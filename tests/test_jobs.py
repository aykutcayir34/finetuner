import time

from finetuner.core.jobs import JobManager


def _wait(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.status in ("pending", "running") and time.time() < deadline:
        time.sleep(0.05)


def test_job_success_and_loss_parsing():
    mgr = JobManager()

    def fake_train(job):
        for step in range(1, 4):
            print(f"step {step}: loss {1.0 / step:.4f}")

    job = mgr.submit("fake", fake_train)
    _wait(job)
    assert job.status == "finished"
    assert len(job.metrics) == 3
    assert job.metrics[0] == (1, 1.0)


def test_job_failure_captured():
    mgr = JobManager()
    job = mgr.submit("boom", lambda j: 1 / 0)
    _wait(job)
    assert job.status == "failed"
    assert "ZeroDivisionError" in job.error


def test_stop_flag():
    mgr = JobManager()

    def loops(job):
        while not job.stop_event.is_set():
            time.sleep(0.01)

    job = mgr.submit("loop", loops)
    time.sleep(0.1)
    mgr.stop(job.id)
    _wait(job)
    assert job.status == "stopped"
