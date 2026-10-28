#!/usr/bin/env python3
"""Verify archived transcript exports with the Inspect AI log reader.

Requires inspect-ai==0.3.279 or a later compatible release.
This is a read-back compatibility check, not an inference or grading run.
"""
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main() -> None:
    for stage in ("dpo", "instruct"):
        source = ROOT / "data" / f"stationhouse_{stage}_pilot_27_unscored.json"
        path = HERE / f"olmo3_identity_spoofing_{stage}_27_archived_unscored.eval"
        original = json.loads(source.read_text(encoding="utf-8"))
        originals = {record["sample_key"]: record for record in original["results"]}
        log = read_eval_log(str(path))
        assert log.status == "success", (stage, log.status)
        assert log.results is not None and log.results.total_samples == 27
        assert log.samples is not None and len(log.samples) == 27
        assert len({str(s.id) for s in log.samples}) == 27
        for item in log.samples:
            recorded = originals[str(item.id)]
            assert item.output is not None
            assert item.target == "", (stage, item.id, item.target)
            assert item.output.completion == recorded["messages"][-1]["content"]
            assert item.scores is None or len(item.scores) == 0
            assert len(item.messages) == len(recorded["messages"])
            for exported, expected in zip(item.messages, recorded["messages"]):
                assert exported.role == expected["role"]
                assert exported.text == expected["content"]
        print(f"Inspect reader verified {stage.upper()}: 27 transcripts, complete messages and final answers, no scores")

if __name__ == "__main__":
    main()
