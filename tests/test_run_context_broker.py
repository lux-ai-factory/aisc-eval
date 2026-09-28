"""The run header on a real broker (adapt plan 2026-09-28, Task 4, review focus 2).

A run_evaluation-like task, sent with the aisc_run header as the backend sends it,
publishes a chain whose last step is a chord callback, like run_evaluation's
workflow (install -> group of plugin chains | finalize). Every task records
run_context.current(); all of them must equal the header that was sent, and a
message sent without the header (standalone) must record None.

Skipped unless AISC_TEST_BROKER_URL is set (and AISC_TEST_RESULT_BACKEND, which the
chord needs, e.g. a throwaway redis). Never point these at a live stack.
"""
import os
import threading

import pytest

BROKER = os.environ.get("AISC_TEST_BROKER_URL")
RESULTS = os.environ.get("AISC_TEST_RESULT_BACKEND")

pytestmark = pytest.mark.skipif(not (BROKER and RESULTS),
                                reason="AISC_TEST_BROKER_URL and AISC_TEST_RESULT_BACKEND not set")

RUN = {"project": "3f2b8c1e-0d4a-4e7b-9a55-1c2d3e4f5a6b",
       "evaluation": "afb49e3f-813d-4260-9919-ee179d1090e6",
       "ticket": "a-ticket-minted-by-the-backend"}


@pytest.fixture
def app_and_seen():
    from celery import Celery, chain, group
    from celery.contrib.testing.worker import start_worker

    from aisc_eval import run_context

    app = Celery("aisc_run_context_broker_test", broker=BROKER, backend=RESULTS)
    app.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, results_extended=True,
                    task_default_queue="aisc-run-context-broker-test")
    seen = []
    done = threading.Event()

    def record(name, request):
        seen.append((name, request.id, run_context.current(), run_context.run_of(request)))

    @app.task(bind=True, name="t.run_evaluation")
    def run_evaluation(self, evaluation_pid):
        record("run_evaluation", self.request)
        workflow = chain(install.si(evaluation_pid),
                         group(plugin.si(evaluation_pid), plugin.si(evaluation_pid))) | finalize.si(evaluation_pid)
        workflow.apply_async()

    @app.task(bind=True, name="t.install")
    def install(self, evaluation_pid):
        record("install", self.request)

    @app.task(bind=True, name="t.plugin")
    def plugin(self, evaluation_pid):
        record("plugin", self.request)

    @app.task(bind=True, name="t.finalize")
    def finalize(self, evaluation_pid):
        record("finalize", self.request)
        done.set()

    @app.task(bind=True, name="t.standalone")
    def standalone(self):
        record("standalone", self.request)

    with start_worker(app, pool="solo", perform_ping_check=False, shutdown_timeout=30):
        yield app, seen, done


def test_every_task_of_the_run_acts_for_the_header_sent(app_and_seen):
    app, seen, done = app_and_seen
    app.send_task("t.standalone").get(timeout=30)
    app.send_task("t.run_evaluation", args=[RUN["evaluation"]], headers={"aisc_run": RUN})
    assert done.wait(60), f"the chord callback never ran; seen: {seen}"

    by_name = {}
    for name, _id, current, header in seen:
        by_name.setdefault(name, []).append((current, header))
    assert by_name["standalone"] == [(None, None)]
    assert [n for n in ("run_evaluation", "install", "plugin", "finalize") if n in by_name] == [
        "run_evaluation", "install", "plugin", "finalize"]
    assert len(by_name["plugin"]) == 2
    for name in ("run_evaluation", "install", "plugin", "finalize"):
        for current, header in by_name[name]:
            assert current == RUN, f"{name} acted for {current!r}"
            assert header == RUN, f"{name} arrived with header {header!r}"
