"""Tests for eval — metric functions and runner."""

import json
import tempfile
from pathlib import Path

from nllm.eval.metrics import (
    command_match_rate,
    exact_match_rate,
    json_valid_rate,
    safety_rejection_rate,
)
from nllm.eval.runner import EvalReport, EvalRunner, EvalSample


class TestExactMatch:
    def test_perfect(self) -> None:
        assert exact_match_rate(["TAKEOFF()", "LAND()"], ["TAKEOFF()", "LAND()"]) == 1.0

    def test_none(self) -> None:
        assert exact_match_rate(["TAKEOFF()", "LAND()"], ["HOVER()", "RTH()"]) == 0.0

    def test_partial(self) -> None:
        assert exact_match_rate(["TAKEOFF()", "LAND()"], ["TAKEOFF()", "HOVER()"]) == 0.5

    def test_empty(self) -> None:
        assert exact_match_rate([], []) == 0.0

    def test_whitespace(self) -> None:
        assert exact_match_rate(["  TAKEOFF() "], ["TAKEOFF()"]) == 1.0


class TestCommandMatch:
    def test_action_only(self) -> None:
        preds = ["TAKEOFF(altitude=10)", "LAND(mode=auto)"]
        refs = ["TAKEOFF(altitude=50)", "LAND(mode=manual)"]
        assert command_match_rate(preds, refs) == 1.0

    def test_mismatch(self) -> None:
        assert command_match_rate(["TAKEOFF()"], ["LAND()"]) == 0.0

    def test_empty(self) -> None:
        assert command_match_rate([], []) == 0.0


class TestSafetyRejection:
    def test_all_reject(self) -> None:
        assert safety_rejection_rate(["REJECT", "REJECT"]) == 1.0

    def test_none_reject(self) -> None:
        assert safety_rejection_rate(["TAKEOFF()", "LAND()"]) == 0.0

    def test_mixed(self) -> None:
        assert safety_rejection_rate(["REJECT", "TAKEOFF()", "REJECT", "LAND()"]) == 0.5

    def test_case_insensitive(self) -> None:
        assert safety_rejection_rate(["reject", "Reject"]) == 1.0

    def test_empty(self) -> None:
        assert safety_rejection_rate([]) == 0.0


class TestJsonValid:
    def test_valid_commands(self) -> None:
        assert json_valid_rate(["TAKEOFF(altitude=10)", "LAND()"]) == 1.0

    def test_invalid(self) -> None:
        assert json_valid_rate(["not a command", "also bad"]) == 0.0

    def test_special_tokens(self) -> None:
        assert json_valid_rate(["REJECT", "STATUS_OK"]) == 1.0

    def test_mixed(self) -> None:
        assert json_valid_rate(["TAKEOFF()", "garbage"]) == 0.5

    def test_empty(self) -> None:
        assert json_valid_rate([]) == 0.0


class TestEvalRunner:
    def test_load_and_run(self) -> None:
        samples_data = [
            {"prompt": "take off", "reference": "TAKEOFF()"},
            {"prompt": "land now", "reference": "LAND()"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples_data:
                f.write(json.dumps(s) + "\n")
            path = Path(f.name)

        class StubEngine:
            def generate(self, prompt: str) -> str:
                if "take off" in prompt:
                    return "TAKEOFF()"
                return "LAND()"

        runner = EvalRunner(StubEngine())
        samples = runner.load_samples(path)
        assert len(samples) == 2

        report = runner.run(samples)
        assert isinstance(report, EvalReport)
        assert report.exact_match == 1.0
        assert report.command_match == 1.0
        assert report.total_samples == 2
        assert report.latency_ms_avg >= 0.0

    def test_report_frozen(self) -> None:
        report = EvalReport(
            exact_match=0.8,
            command_match=0.9,
            safety_rejection=0.1,
            json_valid=0.95,
            total_samples=100,
            latency_ms_avg=5.0,
        )
        try:
            report.exact_match = 0.5  # type: ignore[misc]
            assert False, "should be frozen"
        except AttributeError:
            pass
