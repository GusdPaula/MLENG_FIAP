import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from default_pred.observability.timing import timed
from default_pred.observability.tracker import ExperimentTracker, RunStatus

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Mutable handle the pipeline uses to report how the run ended."""

    status: RunStatus = "success"
    rows: int = 0


@contextmanager
def tracked_run(
    tracker: ExperimentTracker, experiment: str, run_name: str
) -> Iterator[RunOutcome]:
    """Open a tracked run, log key lifecycle events and close it with a status.

    On error the run is closed as failed and the exception is re-raised.
    """
    outcome = RunOutcome()
    tracker.start_run(experiment, run_name)
    logger.info("Run started: experiment=%s run=%s", experiment, run_name)
    try:
        with timed(run_name) as timer:
            yield outcome
    except BaseException:
        outcome.status = "failed"
        logger.exception("Run failed: experiment=%s run=%s", experiment, run_name)
        raise
    finally:
        tracker.log_metrics(
            {
                "duration_ms": timer.elapsed_ms,
                "rows_processed": float(outcome.rows),
                "run_success": 0.0 if outcome.status == "failed" else 1.0,
            }
        )
        tracker.end_run(outcome.status)
        logger.info(
            "Run finished: experiment=%s run=%s status=%s duration_ms=%.2f rows=%d",
            experiment,
            run_name,
            outcome.status,
            timer.elapsed_ms,
            outcome.rows,
        )
