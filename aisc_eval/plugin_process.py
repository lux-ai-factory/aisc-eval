"""How a plugin process is started, and what of its run folder it may touch.

A plugin is code from a catalogue. A worker that runs as root starts each test as a user of its
own, leased from a pool of uids (lease_run_user), with no supplementary groups, so the kernel
refuses it the worker's /proc/<pid>/environ (DJANGO_SECRET_KEY, INTERNAL_API_KEY), every file only
root may read, /app included, and every other test's processes and folders. Its folders
(make_run_folders) are readable by that user only: its home and tmp are its own, its inputs are
root's and readable by its group. The worker then reads what the plugin left in output/ as root,
so that folder must not let the plugin point the worker at anything else.

A plugin starts in a session of its own, so it leads a process group: what it starts (a model
server, a browser) is in that group, and the worker ends the group with the plugin, whether the
plugin ended or the worker stops it.

Installing a plugin runs its code too (a source package builds itself), so it is done as a third
user, the install user, with nothing of the worker's environment. Not the plugin user: the install
cache is the install user's, and a plugin able to write to it would plant a package in the next
run's install. A run's environment is the install user's, so the plugin may run it but not change it.
"""

import contextlib
import fcntl
import os
import pwd
import re
import resource
import shutil
import signal
import stat
import subprocess
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from aisc_eval import plugin_runtime
from aisc_eval.scrub import scrub_text
from aisc_eval.utils import env

#: files a plugin creates are its own
PLUGIN_UMASK = 0o077
#: the setting that names this worker's pool of run uids, "first-last": two worker containers (the run worker and
#: the one that reads manifests) must never lease the same uid, since the kernel counts a uid's processes across
#: containers and a shared volume would show one owner for two runs
RUN_UIDS_SETTING = "PLUGIN_RUN_UIDS"
DEFAULT_RUN_UIDS = range(10100, 10200)
#: the image's own users (10001 plugin, 10002 install, 10003 egress proxy) and below: never a run user
HIGHEST_IMAGE_UID = 10003


def run_uid_pool(value: str | None) -> range:
    """The pool a setting "first-last" names (inclusive), DEFAULT_RUN_UIDS when it is unset or empty. A pool that
    is reversed, reaches the image's own users or is not a range is refused: the worker would not start."""
    if not (value or "").strip():
        return DEFAULT_RUN_UIDS
    first, sep, last = value.partition("-")
    try:
        pool = range(int(first), int(last) + 1)
    except ValueError:
        pool = None
    if not sep or pool is None or len(pool) == 0 or pool.start <= HIGHEST_IMAGE_UID:
        raise ValueError(f"{RUN_UIDS_SETTING} must be first-last, both above {HIGHEST_IMAGE_UID} and first <= "
                         f"last, not {value!r}")
    return pool


#: the uids a root worker runs plugin tests as, one per test at a time: with no passwd entry, so nothing else owns
#: them
RUN_UIDS = run_uid_pool(os.environ.get(RUN_UIDS_SETTING))
#: one lock file per uid; a test holds its uid's lock while it runs (flock, so a holder that died
#: frees it)
RUN_USER_LOCKS = Path("/run/aisc-run-users")


@dataclass(frozen=True)
class PluginUser:
    uid: int
    gid: int
    home: str | None
    #: False when the worker is not root: it cannot change user, so the plugin runs as the worker
    switch: bool


#: what an install sees of the world: the image's own tools, never /app (the worker's code and venv)
INSTALL_PATH = "/usr/local/bin:/usr/bin:/bin"
#: an install's files are read and run by the plugin user
INSTALL_UMASK = 0o022
#: seconds each install step may take: a first-time Python download or a large package is slow
INSTALL_TIMEOUT = 1000


def _user(name: str, role: str) -> PluginUser:
    if os.geteuid() != 0:
        # a worker started by hand (tasks/start_worker.sh) runs plugins as the developer, who
        # can read the worker's environment anyway
        return PluginUser(os.geteuid(), os.getegid(), None, switch=False)
    # no fallback: an image without the user fails the plugin rather than running it as root
    entry = pwd.getpwnam(name)
    if entry.pw_uid == 0:
        raise RuntimeError(f"{role} user {name} is root")
    return PluginUser(entry.pw_uid, entry.pw_gid, entry.pw_dir, switch=True)


def plugin_user(name: str = env.PLUGIN_USER) -> PluginUser:
    """The image's plugin user: kept apart from the install user. A test runs as a leased user
    (lease_run_user)."""
    return _user(name, "plugin")


class NoRunUserFree(RuntimeError):
    """Every uid of the pool is held by a running test."""


@contextlib.contextmanager
def lease_run_user(
    pool: range = RUN_UIDS, locks: Path = RUN_USER_LOCKS
) -> Iterator[PluginUser]:
    """The user one test runs as: the first uid of the pool whose lock is free, held until the test
    ends (or the process holding it dies). Its group is its own (gid equals uid). A worker that is not
    root cannot change user: the plugin runs as the worker, as before."""
    if os.geteuid() != 0:
        yield PluginUser(os.geteuid(), os.getegid(), None, switch=False)
        return
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    for uid in pool:
        lock = open(locks / f"{uid}.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        try:
            yield PluginUser(uid, uid, None, switch=True)
        finally:
            lock.close()
        return
    raise NoRunUserFree(
        f"every plugin user is busy: at most {len(pool)} tests run at once"
    )


def end_processes_of(user: PluginUser) -> None:
    """Kill every process left running as a leased user, before its uid goes to another test: a
    plugin can leave its process group (setsid), and a process left behind would see the next test's
    files. A process of that user sends the kill to everything it may signal, which is that user's
    processes only, all at once, so nothing forks away in between."""
    if not user.switch:
        return
    subprocess.run(
        ["/bin/true"],
        user=user.uid,
        group=user.gid,
        extra_groups=[],
        preexec_fn=_kill_everything_of_this_user,
        check=False,
    )


def _kill_everything_of_this_user() -> None:
    # runs in the child after its switch to the user: kill(-1) skips the caller itself
    with contextlib.suppress(ProcessLookupError):
        os.kill(-1, signal.SIGKILL)


@dataclass(frozen=True)
class RunFolders:
    """One test's folders. root (root's, others may only pass through) holds the rest: work (the
    plugin's working folder, with config.json and runtime/), input and output (in work, where the
    runtime reads and writes them), home and tmp (the plugin user's own) and venv (the install
    user's)."""

    root: Path
    work: Path
    input: Path
    output: Path
    home: Path
    tmp: Path
    venv: Path


def make_run_folders(root: Path, user: PluginUser) -> RunFolders:
    """The folders of one test, each readable only by whom it is for: root:root 0711 for the run
    folder, root:<group> 0750 for work and input (the plugin reads, cannot change), root:<group> 01770
    for output (it adds files; the sticky bit keeps it from replacing others'), <user> 0700 for home
    and tmp. A worker that is not root runs the plugin as itself: the modes alone apply."""
    work = root / "work"
    folders = RunFolders(
        root=root,
        work=work,
        input=work / "input",
        output=work / "output",
        home=root / "home",
        tmp=root / "tmp",
        venv=root / "venv",
    )
    plan = (
        (folders.root, 0o711, None),
        (folders.work, 0o750, (-1, user.gid)),
        (folders.input, 0o750, (-1, user.gid)),
        (folders.output, 0o1770, (-1, user.gid)),
        (folders.home, 0o700, (user.uid, user.gid)),
        (folders.tmp, 0o700, (user.uid, user.gid)),
    )
    for folder, mode, owner in plan:
        folder.mkdir(mode=0o700, exist_ok=True)
        if user.switch and owner is not None:
            os.chown(folder, *owner)
        os.chmod(folder, mode)
    return folders


def write_for_run(path: Path, data: bytes, user: PluginUser) -> None:
    """A file the worker writes for the plugin (config.json, an input, the runtime script): root's,
    readable by the plugin's group only (0640). Created under umask 077, so no other user can read it
    even for the moment before its group is set."""
    previous = os.umask(0o077)
    try:
        path.write_bytes(data)
    finally:
        os.umask(previous)
    if user.switch:
        os.chown(path, 0, user.gid)
    os.chmod(path, 0o640)


def install_user(name: str = env.PLUGIN_INSTALL_USER) -> PluginUser:
    """Who installs plugins: neither root nor the plugin user."""
    user = _user(name, "install")
    if user.switch and user.uid == plugin_user().uid:
        raise RuntimeError(
            f"install user {name} is the plugin user: a plugin could change the install cache"
        )
    return user


def run_install(
    argv: list[str], user: PluginUser, timeout: int
) -> subprocess.CompletedProcess[str]:
    """One step of an install (uv venv, uv pip install) as the install user, in an environment of its own:
    the image's tools, its home and its own uv cache, started in its home. uv would otherwise read the
    configuration of the folder it starts in, the worker's /app, which only root may read. A step runs the
    package's build code, which may leave a process behind (setsid): every process of the install user is
    ended when the step ends, however it ends, as for a test's user."""
    home = user.home or os.path.expanduser("~")
    try:
        return _run_install_step(argv, user, timeout, home)
    finally:
        end_processes_of(user)


def _run_install_step(
    argv: list[str], user: PluginUser, timeout: int, home: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=home,
        env={
            "PATH": INSTALL_PATH,
            "HOME": home,
            "UV_CACHE_DIR": f"{home}/.cache/uv",
            "UV_NO_CONFIG": "1",
        },
        capture_output=True,
        text=True,
        timeout=timeout,
        umask=INSTALL_UMASK,
        user=user.uid if user.switch else None,
        group=user.gid if user.switch else None,
        extra_groups=[] if user.switch else None,
    )


def stage_local_source(source: Path, dest: Path, user: PluginUser) -> Path:
    """A local plugin's folder, copied where the install user may read it: the original is under /app,
    which is root's only. The copy is the install user's."""
    shutil.copytree(source, dest, symlinks=True)
    if user.switch:
        for path in [dest, *dest.rglob("*")]:
            os.lchown(path, user.uid, user.gid)
    return dest


def prepare_install_folder(folder: Path, user: PluginUser) -> None:
    """A folder the install user fills (a venv): its own, which others may read and run but not change. Its
    parent, root's (a run folder before the plugin's group is let in), lets the install user pass through to
    it without listing what else is there."""
    if user.switch:
        parent = folder.parent
        os.chmod(parent, stat.S_IMODE(parent.stat().st_mode) | 0o711)
    folder.mkdir(exist_ok=True)
    if user.switch:
        os.chown(folder, user.uid, user.gid)
    os.chmod(folder, 0o755)


class InstallFailed(RuntimeError):
    """A step of building a plugin's venv failed; the text is the step's scrubbed stderr."""

    def __init__(self, step: str, text: str):
        super().__init__(text)
        #: "venv" (making it) or "install" (filling it)
        self.step = step


@dataclass(frozen=True)
class Pins:
    """What a plugin's environment is installed from: the hash of its wheel (None for a local plugin, which is
    installed from its folder) and the hashed lock of everything it needs."""

    wheel_sha256: str | None
    lock: str


#: the hashed lock of a plugin's dependencies, next to its venv
LOCK_FILE = "requirements.lock"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_hashes(lock: str, package_name: str, version: str) -> list[str]:
    """The sha256 hashes a lock lists for one package at one version (a wheel's and its sdist's)."""
    hashes: list[str] = []
    current = False
    for line in lock.splitlines():
        text = line.strip().rstrip("\\").strip()
        if not text or text.startswith("#"):
            continue
        if not line[:1].isspace():
            name, _sep, pinned = text.split(";")[0].partition("==")
            current = (
                _canonical(name.split("[")[0].strip()) == _canonical(package_name)
                and pinned.strip() == version
            )
        elif current and text.startswith("--hash=sha256:"):
            hashes.append(text.split(":", 1)[1].strip().lower())
    return hashes


def lock_dependencies(
    package_name: str,
    version: str,
    plugin_info: Mapping[str, Any],
    wheel_sha256: str | None,
    folder: Path,
    installer: PluginUser,
    index_url: str | None,
) -> str:
    """The hashed lock of what the plugin needs, compiled by the install user (uv pip compile
    --generate-hashes): a catalogue plugin with itself, whose wheel must be the one recorded (wheel_sha256),
    a local plugin its dependencies only (its own code is installed from its folder). InstallFailed when the
    lock cannot be compiled, or names another wheel than the one recorded."""
    local = plugin_info["source"] == "local"
    if not local and not wheel_sha256:
        raise InstallFailed(
            "lock",
            f"no wheel hash is recorded for {package_name} {version}: a catalogue plugin is installed only from "
            "the wheel recorded for it",
        )
    work = folder / "lock"
    prepare_install_folder(work, installer)
    if local:
        source = stage_local_source(
            plugin_info["pkg_root"].resolve(), work / "source", installer
        )
        requirements = source / "pyproject.toml"
    else:
        requirements = work / "requirements.in"
        requirements.write_text(f"{package_name}=={version}\n")
        os.chmod(requirements, 0o644)
    out = work / LOCK_FILE
    argv = [
        "uv",
        "pip",
        "compile",
        "--generate-hashes",
        "--python-version",
        "3.12",
        "--quiet",
    ]
    if index_url:
        argv.extend(["--extra-index-url", index_url])
    argv.extend([str(requirements), "-o", str(out)])
    compiled = run_install(argv, installer, timeout=INSTALL_TIMEOUT)
    if compiled.returncode != 0:
        raise InstallFailed("lock", scrub_text(compiled.stderr or ""))
    lock = out.read_text()
    if not local and wheel_sha256.lower() not in locked_hashes(
        lock, package_name, version
    ):
        raise InstallFailed(
            "lock",
            f"the index's wheel of {package_name} {version} is not the one recorded "
            f"(sha256 {wheel_sha256[:12]}...)",
        )
    return lock


@dataclass(frozen=True)
class Venv:
    path: Path
    #: the two steps, whose output goes to the run's log
    made: subprocess.CompletedProcess[str]
    installed: subprocess.CompletedProcess[str]

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"


def build_venv(
    package_name: str,
    version: str,
    plugin_info: Mapping[str, Any],
    folder: Path,
    installer: PluginUser,
    index_url: str | None,
    pins: Pins | None = None,
) -> Venv:
    """The plugin installed in folder/venv by the install user: a local plugin from its copy in
    folder/source, a catalogue one from the index (index_url, when there is one, besides PyPI).
    With pins, only what the lock names is installed, each package checked against its hash, and nothing
    it does not name (a local plugin's own code then comes from its folder, without dependencies).
    InstallFailed when a step fails."""
    if plugin_info["source"] == "local":
        install_target = str(
            stage_local_source(
                plugin_info["pkg_root"].resolve(), folder / "source", installer
            )
        )
    else:
        install_target = f"{package_name}=={version}"
    venv = folder / "venv"
    prepare_install_folder(venv, installer)
    made = run_install(
        [
            "uv",
            "venv",
            str(venv),
            "--allow-existing",
            "--python",
            "/usr/local/bin/python",
        ],
        installer,
        timeout=INSTALL_TIMEOUT,
    )
    if made.returncode != 0:
        raise InstallFailed("venv", scrub_text(made.stderr or ""))
    # Online, from the index: the run needs the network anyway (a plugin may download its model),
    # and a plugin's shared dependency resolved from a local workspace source is never cached
    # under the index for an offline resolver. The index is always passed: a probe of it was slow
    # (its root lists the whole PyPI mirror) and dropped it on a timeout, so a catalogue plugin was
    # "not found"; when it is down, uv says so.
    install_cmd = ["uv", "pip", "install", "--python", str(venv)]
    if pins is not None:
        lock = folder / LOCK_FILE
        lock.write_text(pins.lock)
        os.chmod(lock, 0o644)
        install_cmd.extend(["--require-hashes", "--no-deps"])
    if index_url:
        install_cmd.extend(["--extra-index-url", index_url])
    if pins is not None:
        install_cmd.extend(["-r", str(lock)])
    else:
        install_cmd.append(install_target)
    installed = run_install(install_cmd, installer, timeout=INSTALL_TIMEOUT)
    if installed.returncode != 0:
        text = scrub_text(installed.stderr or "")
        if pins is not None and "hash" in text.lower():
            # --require-hashes refused a file whose hash the lock does not list
            text = (
                f"a file on the index for {package_name} {version} or one of its dependencies is not the one "
                f"recorded: {text}"
            )
        raise InstallFailed("install", text)
    if pins is not None and plugin_info["source"] == "local":
        own = run_install(
            ["uv", "pip", "install", "--python", str(venv), "--no-deps", install_target],
            installer,
            timeout=INSTALL_TIMEOUT,
        )
        if own.returncode != 0:
            raise InstallFailed("install", scrub_text(own.stderr or ""))
    return Venv(venv, made, installed)


def freeze(venv: Venv, installer: PluginUser) -> str:
    """What the venv holds, one `name==version` per line (`uv pip freeze`, as the install user): the run's
    environment, kept as a file of the run. InstallFailed when it cannot be read."""
    frozen = run_install(
        ["uv", "pip", "freeze", "--python", str(venv.path)],
        installer,
        timeout=INSTALL_TIMEOUT,
    )
    if frozen.returncode != 0:
        raise InstallFailed("freeze", scrub_text(frozen.stderr or ""))
    return frozen.stdout or ""


def copy_runtime(workspace: Path, user: PluginUser | None = None) -> Path:
    """The runtime script, copied into the run's working folder: the plugin user cannot read /app."""
    folder = workspace / "runtime"
    folder.mkdir()
    if user is None:
        return Path(shutil.copy(plugin_runtime.__file__, folder / "plugin_runtime.py"))
    if user.switch:
        os.chown(folder, -1, user.gid)
    os.chmod(folder, 0o750)
    script = folder / "plugin_runtime.py"
    write_for_run(script, Path(plugin_runtime.__file__).read_bytes(), user)
    return script


#: the settings that cap a plugin's process, with their defaults (MB, or a count); see plugin_limits.
#: Address space is reserved well above what is used: a StrongREJECT process (thread stacks, malloc arenas,
#: an OpenBLAS pool per CPU) reserves about 4 GB to use 300 MB. Its scoring pool starts one process per CPU,
#: about 70 processes and threads per run on a 20-CPU host, about 230 on 64 CPUs. Each test runs as a user of
#: its own, so the process count is per test: four tests at once at their cap (1600) stay under the worker
#: container's 2048 with room for the worker, its children and installs.
#: The largest file leaves room for a local model a plugin downloads; the pilots' own files stay under 10 MB.
LIMIT_SETTINGS = {
    "PLUGIN_MAX_MEMORY_MB": 16384,
    "PLUGIN_MAX_PROCESSES": 400,
    "PLUGIN_MAX_FILE_MB": 8192,
    "PLUGIN_OUTPUT_MAX_MB": 1024,
}
#: seconds a plugin over its CPU time gets between the warning (SIGXCPU) and the kill
CPU_GRACE_S = 60
_MB = 2**20


def _setting(name: str) -> int:
    """A limit's whole-number setting, read when it is used so an operator's change needs no rebuild."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return LIMIT_SETTINGS[name]
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number, not {raw!r}") from None
    if value <= 0:
        raise RuntimeError(f"{name} must be above 0, not {value}")
    return value


def plugin_limits(cpu_seconds: int) -> dict[int, tuple[int, int]]:
    """The (soft, hard) resource limits a plugin runs under.

    Memory is its address space, processes count every process and thread of its user, a file is the
    largest one it may write, and CPU is the processor time it may use before SIGXCPU, then SIGKILL a
    minute later. The worker's time limits still end the test by the clock.
    """
    memory = _setting("PLUGIN_MAX_MEMORY_MB") * _MB
    processes = _setting("PLUGIN_MAX_PROCESSES")
    file_size = _setting("PLUGIN_MAX_FILE_MB") * _MB
    return {
        resource.RLIMIT_AS: (memory, memory),
        resource.RLIMIT_NPROC: (processes, processes),
        resource.RLIMIT_FSIZE: (file_size, file_size),
        resource.RLIMIT_CPU: (cpu_seconds, cpu_seconds + CPU_GRACE_S),
    }


def _apply_limits(limits: Mapping[int, tuple[int, int]]):
    """What the plugin's process runs before the plugin: its limits, after the switch to its user, so
    lowering them needs no privilege."""

    def apply() -> None:
        for limit, (soft, hard) in limits.items():
            resource.setrlimit(limit, (soft, hard))

    return apply


def _user_env(
    plugin_env: dict[str, str], workspace: Path, user: PluginUser
) -> dict[str, str]:
    """A leased user (no home of its own in the image) gets the home, tmp and name of its run: the
    workspace is the run's work folder (make_run_folders), with home/ and tmp/ beside it. A leased uid
    has no passwd entry, and a tool that asks who it is (getpass, git, a model cache under ~) reads
    these."""
    if user.switch and user.home:
        return {**plugin_env, "HOME": user.home}
    if user.switch:
        name = f"aisc-run-{user.uid}"
        return {
            **plugin_env,
            "HOME": str(workspace.parent / "home"),
            "TMPDIR": str(workspace.parent / "tmp"),
            "USER": name,
            "LOGNAME": name,
        }
    return plugin_env


def _user_limits(
    limits: Mapping[int, tuple[int, int]] | None, user: PluginUser
) -> Mapping[int, tuple[int, int]] | None:
    """The process count is a count of the user's processes: a plugin that runs as the worker itself (a
    worker that is not root) would share it with the worker, so it is left out there."""
    if limits and not user.switch:
        return {k: v for k, v in limits.items() if k != resource.RLIMIT_NPROC}
    return limits


def start(
    argv: list[str],
    workspace: Path,
    stderr: IO[str],
    plugin_env: dict[str, str],
    user: PluginUser,
    limits: Mapping[int, tuple[int, int]] | None = None,
) -> subprocess.Popen[str]:
    """Start the plugin under its limits, with its run's home (_user_env); its stdout is the worker's to read
    for progress lines."""
    limits = _user_limits(limits, user)
    return subprocess.Popen(
        argv,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=_user_env(plugin_env, workspace, user),
        text=True,
        bufsize=1,
        umask=PLUGIN_UMASK,
        user=user.uid if user.switch else None,
        group=user.gid if user.switch else None,
        # setgroups([]): root's supplementary groups would otherwise stay with the plugin
        extra_groups=[] if user.switch else None,
        # the plugin leads its own process group, which the worker ends with it
        start_new_session=True,
        preexec_fn=_apply_limits(limits) if limits else None,
    )


class NoAnswer(RuntimeError):
    """A plugin command (its manifest, a form hook) ended without a usable answer; the text says why."""


#: what of a failed command's stderr is kept for its reason
_REASON_BYTES = 64 * 1024


def _drain(stream: IO[bytes], into: bytearray, cap: int) -> None:
    """Read the stream to its end, keeping at most cap + 1 bytes (one more says it was over)."""
    while chunk := stream.read(65536):
        room = cap + 1 - len(into)
        if room > 0:
            into.extend(chunk[:room])


def _feed(stream: IO[bytes], data: bytes) -> None:
    try:
        stream.write(data)
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def run_for_answer(
    argv: list[str],
    workspace: Path,
    plugin_env: dict[str, str],
    user: PluginUser,
    stdin: bytes,
    timeout_s: float,
    max_bytes: int,
    limits: Mapping[int, tuple[int, int]] | None = None,
) -> bytes:
    """Run a short plugin command (read its manifest, run a form hook) as the user, under its limits, and give
    back its stdout. NoAnswer when it does not end within timeout_s (it and everything it started are killed),
    when its stdout is over max_bytes, or when it fails (with the last line of its stderr, scrubbed)."""
    limits = _user_limits(limits, user)
    process = subprocess.Popen(
        argv,
        cwd=str(workspace),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_user_env(plugin_env, workspace, user),
        umask=PLUGIN_UMASK,
        user=user.uid if user.switch else None,
        group=user.gid if user.switch else None,
        extra_groups=[] if user.switch else None,
        start_new_session=True,
        preexec_fn=_apply_limits(limits) if limits else None,
    )
    out, err = bytearray(), bytearray()
    threads = [
        threading.Thread(target=_feed, args=(process.stdin, stdin), daemon=True),
        threading.Thread(target=_drain, args=(process.stdout, out, max_bytes), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, err, _REASON_BYTES), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()
        raise NoAnswer(f"it did not answer within {timeout_s:g} seconds") from None
    finally:
        _signal_group(process, signal.SIGKILL)
        for thread in threads:
            thread.join(timeout=5)
    if len(out) > max_bytes:
        raise NoAnswer(f"its answer is larger than {max_bytes / _MB:g} MiB")
    if returncode != 0:
        lines = scrub_text(err.decode("utf-8", "replace")).strip().splitlines()
        raise NoAnswer(lines[-1][:500] if lines else f"it ended with code {returncode}")
    return bytes(out)


def _signal_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    """Send sig to the plugin's process group; a group with nobody left in it is fine."""
    try:
        os.killpg(process.pid, sig)
    except OSError:
        pass


def wait(process: subprocess.Popen) -> int:
    """The plugin's return code, once it ended; whatever it left running in its group is killed."""
    returncode = process.wait()
    _signal_group(process, signal.SIGKILL)
    return returncode


def terminate(process: subprocess.Popen | None) -> None:
    """Stop the plugin and everything in its group: SIGTERM, then SIGKILL after 5 seconds."""
    if process is None:
        return
    try:
        if process.poll() is None:
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=5)
    except Exception:
        pass


def read_outputs(
    output_dir: Path, user: PluginUser, max_total_bytes: int | None = None
) -> dict[str, bytes]:
    """The files the plugin left in output/, refusing anything but its own regular files.

    The worker reads them as root: a symlink, or a hard link to a file the plugin may not
    read, would otherwise hand the plugin /proc/1/environ or another run's data as an artifact.
    Their total size is checked before any is read (PLUGIN_OUTPUT_MAX_MB by default): the worker
    holds them in memory to upload them.
    """
    if max_total_bytes is None:
        max_total_bytes = _setting("PLUGIN_OUTPUT_MAX_MB") * _MB
    opened: list[tuple[str, int]] = []
    outputs = {}
    folder = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in sorted(os.listdir(folder)):
            try:
                # O_NONBLOCK: opening a FIFO would otherwise hold the worker until someone writes
                fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Plugin output {name} is not a regular file of the plugin's"
                ) from exc
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != user.uid
                or info.st_nlink != 1
            ):
                os.close(fd)
                raise RuntimeError(
                    f"Plugin output {name} is not a regular file of the plugin's"
                )
            opened.append((name, fd))
        total = sum(os.fstat(fd).st_size for _name, fd in opened)
        if total > max_total_bytes:
            raise RuntimeError(
                f"The plugin's output files are larger than {max_total_bytes / _MB:g} MB in total"
            )
        while opened:
            name, fd = opened.pop(0)
            with os.fdopen(fd, "rb") as file:
                outputs[name] = file.read()
    finally:
        for _name, fd in opened:
            os.close(fd)
        os.close(folder)
    return outputs
