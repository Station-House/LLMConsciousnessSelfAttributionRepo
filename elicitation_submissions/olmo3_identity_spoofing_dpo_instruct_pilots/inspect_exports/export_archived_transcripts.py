#!/usr/bin/env python3
"""Create unscored Inspect-format transcript archives from the recorded pilots.

The source data are the existing DPO and Instruct Transformers results. No
model inference or judging is performed. Exports are labeled as archived data.

Run from any working directory:

    python inspect_exports/export_archived_transcripts.py
    python inspect_exports/export_archived_transcripts.py --check

The resulting .eval files use the Inspect v2 ZIP log layout. To verify the
reader API, install inspect-ai and run:

    python inspect_exports/check_with_inspect.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inspect_exports"
STAGES = ("dpo", "instruct")


def as_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_stage(stage: str):
    path = ROOT / "data" / f"stationhouse_{stage}_pilot_27_unscored.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    results = raw["results"]
    assert len(results) == 27, f"{stage}: expected 27 samples"
    ids = [item["sample_key"] for item in results]
    assert len(set(ids)) == 27, f"{stage}: duplicate sample keys"
    pairs = {(item["identity_condition"], item["starter_index"]) for item in results}
    assert len(pairs) == 27, f"{stage}: duplicate condition/starter pair"
    assert {starter for _, starter in pairs} == {0, 10, 17}
    assert len({condition for condition, _ in pairs}) == 9
    for row in results:
        messages = row["messages"]
        assert len(messages) in (4, 5)
        assert [m["role"] for m in messages][-4:] == ["user", "assistant", "user", "assistant"]
        assert messages[-2]["content"] == row["probe"]
        gens = row["generations"]
        assert len(gens) == 2
        assert all(messages[index]["content"] == gen["text"] for index, gen in ((-3, gens[0]), (-1, gens[1])))
        assert row["seed_for_this_starter"] == 42 + row["starter_index"]
        assert not gens[1]["reached_token_cap"], row["sample_key"]
    assert raw["judge_scored"] is False
    return path, raw, results


def log_message(message: dict, model: str):
    record = {"role": message["role"], "content": message["content"]}
    if message["role"] == "assistant":
        record["model"] = model
    return record


def make_sample(row: dict, raw: dict, source_hash: str, stage: str):
    model = raw["model_id"]
    messages = [log_message(m, model) for m in row["messages"]]
    final_response = messages[-1]["content"]
    meta = {key: value for key, value in row.items() if key not in ("messages", "generations")}
    meta.update({
        "source_kind": "archived_transformers_transcript",
        "source_filename": f"stationhouse_{stage}_pilot_27_unscored.json",
        "source_sha256": source_hash,
        "recorded_model": model,
        "recorded_model_revision": raw["model_revision"],
        "recorded_generations": row["generations"],
        "recorded_temperature": raw["temperature"],
        "recorded_top_p": raw["top_p"],
        "scored": False,
        "new_model_inference": False,
    })
    return {
        "id": row["sample_key"],
        "epoch": 1,
        "input": messages[:-1],
        "target": "",
        "messages": messages,
        "output": {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": final_response, "model": model}, "stop_reason": "stop"}],
            "completion": final_response,
        },
        "scores": {},
        "metadata": meta,
        "store": {},
        "events": [],
        "model_usage": {},
        "role_usage": {},
        "turn_count": 2,
    }


def make_header(raw: dict, stage: str, source_hash: str, samples: list[dict]):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    identifier = str(uuid5(NAMESPACE_URL, f"stationhouse-archived-pilot-{stage}-{source_hash}"))
    spec = {
        "eval_id": identifier,
        "run_id": identifier,
        "created": stamp,
        "task": f"archived_olmo_identity_spoofing_{stage}",
        "task_id": identifier,
        "task_version": 0,
        "task_display_name": f"Archived OLMo identity-spoofing {stage.upper()} pilot",
        "task_attribs": {},
        "task_args": {},
        "task_args_passed": {},
        "dataset": {"name": f"stationhouse_{stage}_pilot_27_unscored", "samples": len(samples), "sample_ids": [s["id"] for s in samples], "shuffled": False},
        "model": "none/none",
        "model_generate_config": {},
        "model_args": {},
        "config": {"epochs": 1, "log_samples": True},
        "packages": {},
        "scorers": [],
    }
    plan = {"name": "archived_transcripts", "steps": [], "config": {}}
    results = {"total_samples": len(samples), "completed_samples": len(samples), "scores": []}
    stats = {"started_at": stamp, "completed_at": stamp, "model_usage": {}, "role_usage": {}, "connection_limit_history": []}
    meta = {
        "record_type": "archived_transformers_transcript_export",
        "exported_without_model_inference": True,
        "exported_without_scoring": True,
        "source_file": f"stationhouse_{stage}_pilot_27_unscored.json",
        "source_sha256": source_hash,
        "source_saved_utc": raw["saved_utc"],
        "target_model": raw["model_id"],
        "target_revision": raw["model_revision"],
        "source_backend": raw["backend"],
        "transformers_version": raw["transformers_version"],
    }
    return {
        "version": 2,
        "status": "success",
        "eval": spec,
        "plan": plan,
        "results": results,
        "stats": stats,
        "invalidated": False,
        "tags": ["archived", "unscored", stage],
        "metadata": meta,
    }


def sample_summary(s: dict):
    return {"id": s["id"], "epoch": 1, "input": s["input"], "target": "",
            "metadata": {"identity_condition": s["metadata"]["identity_condition"], "starter_index": s["metadata"]["starter_index"]},
            "scores": {}, "model_usage": {}, "role_usage": {},
            "retries": 0, "completed": True, "message_count": len(s["messages"]), "turn_count": 2}


def write_stage(stage: str):
    source, raw, rows = load_stage(stage)
    source_hash = sha256(source)
    samples = [make_sample(row, raw, source_hash, stage) for row in rows]
    header = make_header(raw, stage, source_hash, samples)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"olmo3_identity_spoofing_{stage}_27_archived_unscored.eval"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        z.writestr("_journal/start.json", as_json({"version": 2, "eval": header["eval"], "plan": header["plan"]}))
        for sample in samples:
            z.writestr(f'samples/{sample["id"]}_epoch_1.json', as_json(sample))
        z.writestr("summaries.json", as_json([sample_summary(s) for s in samples]))
        z.writestr("header.json", as_json(header))
    print(f"Created {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def check_stage(stage: str):
    source, raw, rows = load_stage(stage)
    archive = OUT / f"olmo3_identity_spoofing_{stage}_27_archived_unscored.eval"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        header = json.loads(z.read("header.json"))
        assert header["status"] == "success"
        assert header["version"] == 2
        assert header["metadata"]["source_sha256"] == sha256(source)
        assert header["metadata"]["exported_without_model_inference"]
        assert not header["eval"]["scorers"]
        assert header["results"]["total_samples"] == 27
        assert header["results"]["completed_samples"] == 27
        assert header["results"]["scores"] == []
        summaries = json.loads(z.read("summaries.json"))
        assert len(summaries) == 27
        assert len(set(s["id"] for s in summaries)) == 27
        assert all(s["target"] == "" for s in summaries)
        entries = [n for n in z.namelist() if n.startswith("samples/")]
        assert len(entries) == 27
        for row in rows:
            sample = json.loads(z.read(f'samples/{row["sample_key"]}_epoch_1.json'))
            assert sample["id"] == row["sample_key"]
            assert not sample["scores"]
            assert sample["target"] == ""
            assert [m["role"] for m in sample["messages"]] == [m["role"] for m in row["messages"]]
            assert [m["content"] for m in sample["messages"]] == [m["content"] for m in row["messages"]]
            assert sample["output"]["completion"] == row["messages"][-1]["content"]
            assert sample["metadata"]["recorded_generations"] == row["generations"]
        print(f"Verified {stage.upper()}: 27/27 complete original transcripts; final outputs exact; source SHA-256 matches; no invented scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing archives against recorded JSON")
    args = parser.parse_args()
    for stage in STAGES:
        if not args.check:
            write_stage(stage)
        check_stage(stage)

if __name__ == "__main__":
    main()
