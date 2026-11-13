import logging

import pytest

from default_pred.observability.timing import timed


def test_timed_measures_positive_duration_and_logs(caplog):
    with caplog.at_level(logging.INFO):
        with timed("step") as timer:
            sum(range(1000))

    assert timer.elapsed_ms > 0
    assert any("step finished" in record.message for record in caplog.records)


def test_timed_records_duration_even_when_block_raises(caplog):
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            with timed("boom") as timer:
                raise RuntimeError("fail")

    assert timer.elapsed_ms >= 0
    assert any("boom finished" in record.message for record in caplog.records)
