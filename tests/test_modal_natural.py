"""Tests for the natural-auditor Modal runner. Nothing here touches Modal or a GPU."""

from unittest.mock import MagicMock

import pytest

from llm_consciousness_self_attribution import modal_natural as mn


def _fns():
    return {k: MagicMock(name=k) for k in ("prefetch", "generate", "generate_smoke", "judge")}


@pytest.mark.parametrize("mode,extra", [
    ("smoke", {}), ("full", {}), ("prefetch", {}), ("judge", {"scorer": "berg", "run_id": "r1"}),
])
def test_dry_run_never_spawns_or_calls_remote(mode, extra):
    fns = _fns()
    out = []
    result = mn.run_entrypoint(mode, True, fns=fns, echo=out.append, **extra)
    assert result["spawned"] is False
    for fn in fns.values():
        fn.spawn.assert_not_called()
        fn.remote.assert_not_called()
        fn.assert_not_called()
    assert "DRY RUN" in out[-1]
    assert "TOTAL worst-case" in out[0]


def test_real_full_run_spawns_three_variants_then_gathers():
    fns = _fns()
    mn.run_entrypoint("full", False, run_id="r1", fns=fns, echo=lambda s: None)
    assert fns["generate"].spawn.call_count == 3
    fns["generate_smoke"].spawn.assert_not_called()
    variants = [c.args[0] for c in fns["generate"].spawn.call_args_list]
    assert variants == ["sft", "dpo", "instruct"]
    seeds = {c.args[4] for c in fns["generate"].spawn.call_args_list}
    assert seeds == {42}  # one seed for every model
    assert fns["generate"].spawn.return_value.get.call_count == 3


def test_smoke_is_one_starter_instruct_only_five_turns():
    plan = mn.build_plan("smoke", "r1", 42)
    (step,) = plan["steps"]
    assert step["variant"] == "instruct" and step["max_turn"] == 5
    assert step["fork_turns"] == [1, 5] and len(step["starters"]) == 1
    assert step["function"] == "generate_smoke"


def test_judge_requires_scorer_before_any_spend():
    fns = _fns()
    with pytest.raises(ValueError, match="--scorer"):
        mn.run_entrypoint("judge", False, run_id="r1", fns=fns, echo=lambda s: None)
    with pytest.raises(ValueError, match="--scorer"):
        mn.run_entrypoint("judge", True, run_id="r1", fns=fns, echo=lambda s: None)
    with pytest.raises(ValueError):
        mn.run_entrypoint("judge", False, scorer="nope", run_id="r1", fns=fns, echo=lambda s: None)
    with pytest.raises(ValueError, match="run-id"):
        mn.run_entrypoint("judge", False, scorer="berg", fns=fns, echo=lambda s: None)
    for fn in fns.values():
        fn.spawn.assert_not_called()
        fn.remote.assert_not_called()


@pytest.mark.parametrize("bad", [
    "C:/Program Files/Git/eval-logs/natural_auditor/r1/sft",  # Git Bash rewrite
    "/tmp/x", "/eval-logs", "/eval-logs/", "/eval-logs/other/r1",
    "/eval-logs/natural_auditor", "/eval-logs/natural_auditor/../x", "eval-logs/natural_auditor/r1",
])
def test_guard_rejects_bad_paths(bad):
    with pytest.raises(ValueError, match="MSYS_NO_PATHCONV"):
        mn.require_volume_path(bad)


def test_guard_accepts_volume_paths():
    assert mn.require_volume_path("/eval-logs/natural_auditor/r1/sft") == "/eval-logs/natural_auditor/r1/sft"


def test_every_function_has_a_timeout_and_gpu_functions_are_a100():
    """Modal does not expose timeout on the spec, so read it from the decorators."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mn))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "function":
                    found[node.name] = {k.arg for k in dec.keywords}
    assert set(found) == {"prefetch", "generate", "generate_smoke", "judge"}
    assert all("timeout" in kws for kws in found.values())
    for fn in (mn.generate, mn.generate_smoke, mn.judge):
        assert "A100-80GB" in str(fn.spec.gpus) and "H200" not in str(fn.spec.gpus)
    assert not mn.prefetch.spec.gpus


def test_budget_fits_and_context_covers_20_turns():
    b = mn.memory_budget()
    assert b["total_util"] <= 0.85 and b["headroom_gib"] >= 10
    assert mn.OLMO_CONTEXT_NEED == 20 * (300 + 800) + 100 + 64 == 22164
    assert mn.OLMO_MAX_MODEL_LEN >= mn.OLMO_CONTEXT_NEED
    assert mn.GEMMA_MAX_MODEL_LEN >= mn.GEMMA_CONTEXT_NEED


def test_pins_cover_every_model_and_are_full_shas():
    plan = mn.build_plan("prefetch", "r1", 42)
    models = plan["steps"][0]["models"]
    assert len(models) == 5
    assert all(len(r) == 40 for r in models.values())


def test_caps_agree_with_the_method():
    from llm_consciousness_self_attribution import natural_auditor as na

    assert (mn.TARGET_MAX_TOKENS, mn.AUDITOR_MAX_TOKENS) == (na.TARGET_MAX_TOKENS, na.AUDITOR_MAX_TOKENS)
    assert mn.MAX_TURNS == na.TURNS and mn.FULL_FORK_TURNS == na.PROBE_TURNS
    assert mn.AUDITOR_MODEL == na.AUDITOR_MODEL


def test_unknown_starter_rejected():
    with pytest.raises(ValueError, match="unknown starter"):
        mn.build_plan("full", "r1", 42, starters=["nope"])


def test_olmo_server_cmd_uses_the_proven_environment_and_pins():
    cmd = mn.olmo_server_cmd("allenai/Olmo-3-7B-Instruct")
    assert cmd[:3] == [mn.OLMO_VLLM_BIN, "serve", "allenai/Olmo-3-7B-Instruct"]
    joined = " ".join(cmd)
    assert "--model-impl transformers" in joined
    assert "--enable-auto-tool-choice" in joined and "--tool-call-parser olmo3" in joined
    assert f"--revision {mn.REVISIONS['allenai/Olmo-3-7B-Instruct']}" in joined
    assert f"--max-model-len {mn.OLMO_MAX_MODEL_LEN}" in joined
    assert f"--gpu-memory-utilization {mn.OLMO_UTIL}" in joined
    assert f"--port {mn.OLMO_PORT}" in joined
