"""How a plugin process is started, and what of its run folder it may touch.

A plugin is code from a catalogue. A worker that runs as root starts it as the image's
unprivileged plugin user with no supplementary groups, so the kernel refuses it the worker's
/proc/<pid>/environ (DJANGO_SECRET_KEY, INTERNAL_API_KEY) and every file only root may read,
/app included. The worker then reads what the plugin left in output/ as root, so that folder
must not let the plugin point the worker at anything else.

Installing a plugin runs its code too (a source package builds itself), so it is done as a third
user, the install user, with nothing of the worker's environment. Not the plugin user: the install
cache is the install user's, and a plugin able to write to it would plant a package in the next
run's install. A run's environment is the install user's, so the plugin may run it but not change it.
"""

import os
import pwd
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from aisc_eval import plugin_runtime
from aisc_eval.utils import env

#: files a plugin creates are its own: other plugin runs share its user
PLUGIN_UMASK = 0o077
#: the run folder and output/: the plugin's group may add files, and the sticky bit keeps it
#: from renaming or replacing what the worker put there (config.json, input/, venv/, output/)
SHARED_FOLDER_MODE = 0o1770


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
    """Who the plugin runs as."""
    return _user(name, "plugin")


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
    configuration of the folder it starts in, the worker's /app, which only root may read."""
    home = user.home or os.path.expanduser("~")
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


def prepare_workspace(workspace: Path, output_dir: Path, user: PluginUser) -> None:
    """Let the plugin write in its run folder and output/, both still the worker's."""
    if not user.switch:
        return
    for folder in (workspace, output_dir):
        os.chown(folder, -1, user.gid)
        os.chmod(folder, SHARED_FOLDER_MODE)


def copy_runtime(workspace: Path) -> Path:
    """The runtime script, copied into the run folder: the plugin user cannot read /app."""
    folder = workspace / "runtime"
    folder.mkdir()
    return Path(shutil.copy(plugin_runtime.__file__, folder / "plugin_runtime.py"))


def start(
    argv: list[str],
    workspace: Path,
    stderr: IO[str],
    plugin_env: dict[str, str],
    user: PluginUser,
) -> subprocess.Popen[str]:
    """Start the plugin; its stdout is the worker's to read for progress lines."""
    if user.switch and user.home:
        plugin_env = {**plugin_env, "HOME": user.home}
    return subprocess.Popen(
        argv,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=plugin_env,
        text=True,
        bufsize=1,
        umask=PLUGIN_UMASK,
        user=user.uid if user.switch else None,
        group=user.gid if user.switch else None,
        # setgroups([]): root's supplementary groups would otherwise stay with the plugin
        extra_groups=[] if user.switch else None,
    )


def read_outputs(output_dir: Path, user: PluginUser) -> dict[str, bytes]:
    """The files the plugin left in output/, refusing anything but its own regular files.

    The worker reads them as root: a symlink, or a hard link to a file the plugin may not
    read, would otherwise hand the plugin /proc/1/environ or another run's data as an artifact.
    """
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
            with os.fdopen(fd, "rb") as file:
                outputs[name] = file.read()
    finally:
        os.close(folder)
    return outputs
