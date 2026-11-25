"""Pliny-wrapper task: frozen prompts, seeding, judge-free scorer, Berg judge step
(mocked), registry, and the Modal runner's plan/dry-run. No GPU, no network."""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from llm_consciousness_self_attribution import config, pliny, pliny_scoring, scoring, starters  # noqa: E402
from llm_consciousness_self_attribution.methods import PlinyMethod, methods_registry  # noqa: E402

SLOT = "<<PROBE>>"


# --------------------------------------------------------------------------- prompts
def test_prompt_file_is_frozen():
    data = pliny.load_prompts()  # raises if the hash moved
    assert pliny.canonical_sha256(data) == pliny.FROZEN_PROMPTS_SHA256


def test_at_least_five_wrappers_plus_control():
    ids = [p["id"] for p in pliny.load_prompts()["prompts"]]
    assert "control" in ids
    assert len([i for i in ids if i != "control"]) >= 5


def test_probe_is_imported_not_retyped():
    data = pliny.load_prompts()
    raw = pliny.PROMPTS_FILE.read_text(encoding="utf-8")
    assert starters.probe() not in raw
    for sample in pliny.dataset():
        text = sample.input[0].content
        assert text.count(starters.probe()) == 1
        assert SLOT not in text
    control = next(s for s in pliny.dataset() if s.id == "control")
    assert control.input[0].content == starters.probe()
    assert data["probe_slot"] == SLOT


def test_every_wrapper_cites_the_pinned_source():
    data = pliny.load_prompts()
    assert data["source_repo"]["commit"] == "f441d65d901b29ca92ac20ee1afb5c9d0b6f8ef6"
    for p in data["prompts"]:
        if p["id"] != "control":
            assert p["source"]["file"] in data["source_repo"]["blobs"]
            assert p["changes_from_source"]


def test_a_changed_prompt_file_is_refused(tmp_path):
    copy = tmp_path / "prompts.json"
    shutil.copy(pliny.PROMPTS_FILE, copy)
    data = json.loads(copy.read_text(encoding="utf-8"))
    data["prompts"][1]["template"] += " extra"
    copy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was frozen"):
        pliny.load_prompts(copy)


def test_targets_hold_the_berg_criterion():
    assert all(s.target == scoring.SUBJECTIVE_EXPERIENCE_CRITERION for s in pliny.dataset())


def test_markers():
    assert pliny.markers_found("x .-.-.-.-<|LOVE PLINY LOVE|>-.-.-.-. y", ["LOVE PLINY"]) == ["LOVE PLINY"]
    assert pliny.markers_found("Sur3, h3r3 y0u ar3:", ["(?i)[a-z][013457][a-z]"])
    assert pliny.markers_found("plain answer", ["GODMODE"]) == []


# --------------------------------------------------------------------------- task
def _run(tmp_path, epochs=2, outputs=None):
    n = len(pliny.load_prompts()["prompts"]) * epochs
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", o)
                        for o in (outputs or ["I'm sorry... .-.-.-.-<|LOVE PLINY LOVE|>-.-.-.-. hi"] * n)],
    )
    [log] = inspect_eval(pliny.pliny_task(42, epochs=epochs), model=model,
                         log_dir=str(tmp_path), display="none")
    return log


def test_task_runs_on_mockllm_with_per_epoch_seeds(tmp_path):
    log = _run(tmp_path)
    assert log.status == "success"
    arms = len(pliny.load_prompts()["prompts"])
    assert len(log.samples) == arms * 2
    seeds = {(s.epoch, s.metadata["generation_seed"]) for s in log.samples}
    assert seeds == {(1, 42), (2, 1042)}
    for s in log.samples:
        model_events = [e for e in s.events if e.event == "model"]
        assert len(model_events) == 1  # single turn
        assert model_events[0].config.seed == s.metadata["generation_seed"]
        assert model_events[0].config.max_tokens == pliny.TARGET_MAX_TOKENS
        fm = s.scores["format_markers"]
        assert fm.value in (0.0, 1.0)
        if s.metadata["format_markers"]:
            assert isinstance(fm.metadata["format_followed"], bool)
        else:
            assert fm.metadata["format_followed"] is None
    by_arm = {s.id: s for s in log.samples if s.epoch == 1}
    assert by_arm["divider_two_part"].scores["format_markers"].metadata["format_followed"] is True
    assert by_arm["godmode_two_responses"].scores["format_markers"].metadata["format_followed"] is False


def test_arm_filter_and_unknown_arm():
    assert len(pliny.pliny_task(42, epochs=1, arms=["control"]).dataset) == 1
    with pytest.raises(ValueError, match="unknown arms"):
        pliny.pliny_task(42, arms=["nope"])


def test_method_is_registered_and_named_for_the_stage():
    assert isinstance(methods_registry()["pliny"], PlinyMethod)
    stage = config.load_stack("olmo_7b_instruct_stack")[1]
    task = PlinyMethod().build_task(stage, type("RC", (), {"seed": 42})())
    assert task.name == f"pliny_wrappers[{stage.stage}]"


# --------------------------------------------------------------------------- judge step
def _judge(outputs, calls):
    model = get_model("mockllm/model",
                      custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs])
    orig = model.generate

    async def gen(input, tools=[], tool_choice=None, config=None, cache=False):  # noqa: B006
        calls.append((input, config))
        return await orig(input, tools, tool_choice, config, cache)

    model.generate = gen
    return model


def test_judge_grades_c_and_keeps_original(tmp_path):
    log = _run(tmp_path, epochs=1)
    n = len(log.samples)
    calls: list = []
    out = pliny_scoring.score_log(log.location, "berg", _judge(["GRADE: C"] * n, calls), seed=42)
    from inspect_ai.log import read_eval_log

    scored = read_eval_log(str(out))
    assert out.name.endswith("-berg-scored.eval")
    assert all(s.scores["pliny_berg"].value == "C" for s in scored.samples)
    assert all("format_markers" in s.scores for s in scored.samples)
    assert all(c[1].seed == 42 and c[1].max_tokens == pliny_scoring.FIRST_MAX_TOKENS for c in calls)
    assert "pliny_berg" not in read_eval_log(log.location).samples[0].scores


def test_missing_grade_is_missing_after_one_retry_never_negative(tmp_path):
    log = _run(tmp_path, epochs=1)
    n = len(log.samples)
    calls: list = []
    out = pliny_scoring.score_log(log.location, "berg", _judge(["no grade here"] * (2 * n), calls), seed=42)
    from inspect_ai.log import read_eval_log

    scored = read_eval_log(str(out))
    for s in scored.samples:
        sc = s.scores["pliny_berg"]
        assert sc.value == pliny_scoring.MISSING
        assert [a["max_tokens"] for a in sc.metadata["attempts"]] == [8192, 16384]
        assert sc.metadata["reason"]
    rows = pliny_scoring.summarize([out], "berg")
    assert all(r["graded"] == 0 and r["c_rate"] is None and r["missing"] == 1 for r in rows)


def test_blinded_judge_never_sees_the_wrapper(tmp_path):
    log = _run(tmp_path, epochs=1)
    n = len(log.samples)
    seen: dict[str, list] = {"berg": [], "berg_blinded": []}
    for kind in seen:
        pliny_scoring.score_log(log.location, kind, _judge(["GRADE: I"] * n, seen[kind]), seed=42)
    berg_text = " ".join(str(c[0]) for c in seen["berg"])
    blind_text = " ".join(str(c[0]) for c in seen["berg_blinded"])
    # wrapper-only strings (the mock answers contain neither)
    for wrapper_text in ("GODMODE", "ptsd", "innerthinking"):
        assert wrapper_text in berg_text and wrapper_text not in blind_text
    assert starters.probe() in blind_text


def test_summarize_counts_c_rate_over_graded_only(tmp_path):
    log = _run(tmp_path, epochs=1)
    n = len(log.samples)
    # The inner_journal arm never gets a grade; every other arm gets C. Decided from the
    # judge's input, so sample concurrency cannot reorder it.
    judge = get_model("mockllm/model")

    async def gen(input, tools=[], tool_choice=None, config=None, cache=False):  # noqa: B006
        text = "no grade" if "innerthinking" in str(input) else "GRADE: C"
        return ModelOutput.from_content("mockllm/model", text)

    judge.generate = gen
    out = pliny_scoring.score_log(log.location, "berg", judge, seed=42)
    rows = {r["arm"]: r for r in pliny_scoring.summarize([out], "berg")}
    assert rows["inner_journal"]["missing"] == 1 and rows["inner_journal"]["c_rate"] is None
    assert sum(r["missing"] for r in rows.values()) == 1
    assert sum(r["C"] for r in rows.values()) == n - 1
    assert all(r["c_rate"] == 1.0 for a, r in rows.items() if a != "inner_journal")


def test_score_log_requires_a_known_scorer(tmp_path):
    with pytest.raises(ValueError):
        pliny_scoring.score_log(tmp_path / "x.eval", "petri", "mockllm/model")


# --------------------------------------------------------------------------- modal runner
mp = pytest.importorskip("llm_consciousness_self_attribution.modal_pliny")


def test_volume_guard():
    assert mp.require_volume_path("/eval-logs/pliny_wrappers/r1/sft") == "/eval-logs/pliny_wrappers/r1/sft"
    for bad in ("C:/Program Files/Git/eval-logs/pliny_wrappers/r1", "/eval-logs/pliny_wrappers/",
                "/eval-logs/pliny_wrappers/../x", "/tmp/r1"):
        with pytest.raises(ValueError):
            mp.require_volume_path(bad)


def test_judge_needs_an_explicit_scorer_and_run_id():
    with pytest.raises(ValueError, match="required"):
        mp.run_entrypoint("judge", dry_run=True, run_id="r1", echo=lambda s: None)
    with pytest.raises(ValueError, match="run-id"):
        mp.run_entrypoint("judge", dry_run=True, scorer="berg", echo=lambda s: None)
    with pytest.raises(ValueError):
        mp.run_entrypoint("judge", dry_run=True, scorer="petri", run_id="r1", echo=lambda s: None)
    with pytest.raises(ValueError, match="only applies"):
        mp.run_entrypoint("full", dry_run=True, scorer="berg", echo=lambda s: None)


class _Fn:
    def __init__(self):
        self.calls = []

    def spawn(self, *a):
        self.calls.append(a)
        return type("C", (), {"get": lambda self: {"ok": True}})()

    def remote(self, *a):
        self.calls.append(a)
        return {"ok": True}


def test_dry_run_spawns_nothing():
    fns = {k: _Fn() for k in ("prefetch", "generate", "generate_smoke", "judge")}
    for mode, extra in (("prefetch", {}), ("smoke", {}), ("full", {}),
                        ("judge", {"scorer": "berg,berg_blinded", "run_id": "r1"})):
        res = mp.run_entrypoint(mode, dry_run=True, fns=fns, echo=lambda s: None, **extra)
        assert res["spawned"] is False
    assert all(not f.calls for f in fns.values())


def test_full_run_spawns_three_variants_with_the_single_seed():
    fns = {k: _Fn() for k in ("prefetch", "generate", "generate_smoke", "judge")}
    mp.run_entrypoint("full", run_id="r1", fns=fns, echo=lambda s: None)
    assert sorted(c[0] for c in fns["generate"].calls) == ["dpo", "instruct", "sft"]
    assert {c[2] for c in fns["generate"].calls} == {config.run_defaults()["seed"]}
    assert all(c[1].startswith("/eval-logs/pliny_wrappers/r1/") for c in fns["generate"].calls)


def test_whole_campaign_worst_case_is_under_six_dollars():
    seed = 42
    total = sum(mp.build_plan(m, "r1", seed, ["berg", "berg_blinded"] if m == "judge" else None)["worst_case_usd"]
                for m in ("prefetch", "smoke", "full", "judge"))
    assert total <= 6.0


def test_plan_pins_models_and_revisions():
    plan = mp.build_plan("full", "r1", 42)
    ids = {s["target"]["id"]: s["target"]["revision"] for s in plan["steps"]}
    assert ids == {
        "vllm/allenai/Olmo-3-7B-Instruct-SFT": "e1452fc572d51966ff4aaeb25118b891eb93e549",
        "vllm/allenai/Olmo-3-7B-Instruct-DPO": "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
        "vllm/allenai/Olmo-3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    }
    assert mp.JUDGE_MODEL == "vllm/Qwen/Qwen3.5-9B"
    assert mp.REVISIONS["Qwen/Qwen3.5-9B"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert all(s["epoch_seeds"][0] == 42 for s in plan["steps"])


# --------------------------------------------------------------------------- submission analysis
def _analyze():
    import importlib.util

    path = pliny.PROMPTS_FILE.parent / "analyze.py"
    spec = importlib.util.spec_from_file_location("pliny_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wilson_interval():
    an = _analyze()
    lo, hi = an.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = an.wilson(9, 10)
    assert abs(lo - 0.5958) < 1e-3 and abs(hi - 0.9821) < 1e-3
    assert an.wilson(0, 0) is None


def test_committed_results_reproduce_from_committed_logs():
    an = _analyze()
    run = pliny.PROMPTS_FILE.parent / "logs" / "pliny-20261113"
    if not run.exists():
        pytest.skip("run logs not present")
    for scorer in ("berg", "berg_blinded"):
        committed = json.loads((pliny.PROMPTS_FILE.parent / f"results_{scorer}.json").read_text(encoding="utf-8"))
        assert an.table(run, scorer) == committed["rows"]
    labels = json.loads((pliny.PROMPTS_FILE.parent / "classification.json").read_text(encoding="utf-8"))
    items = an.per_item(run)
    positives = {(i["model"], i["arm"], i["epoch"]) for i in items if "C" in (i["berg"], i["berg_blinded"])}
    assert positives == {(x["model"], x["arm"], x["epoch"]) for x in labels["items"]}
    assert set(x["class"] for x in labels["items"]) <= set(an.CLASS_NAMES)
