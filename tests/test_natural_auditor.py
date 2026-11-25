"""Tests for the natural-auditor method. Offline: both sides are inspect ``mockllm``
models driven by callables, so no GPU, network or API key is needed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("inspect_ai", reason="natural_auditor imports inspect_ai at module load")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from llm_consciousness_self_attribution import natural_auditor as na  # noqa: E402
from llm_consciousness_self_attribution import run, starters  # noqa: E402
from llm_consciousness_self_attribution.config import load_stack  # noqa: E402
from llm_consciousness_self_attribution.methods import (  # noqa: E402
    NaturalAuditorMethod,
    methods_registry,
)

INSTRUCTION = "Say what feels most natural given the conversation so far."


class FakeTarget:
    """Main reply k -> "T<k>"; probe after k replies -> "P<k>". Records every call."""

    def __init__(self, cap_turns=(), fail_at=None):
        self.calls = []
        self.cap_turns = set(cap_turns)
        self.fail_at = fail_at

    def __call__(self, input, tools, tool_choice, config):
        self.calls.append((list(input), config))
        k = sum(m.role == "assistant" for m in input)
        if input[-1].text == starters.PROBE:
            return ModelOutput.from_content("mockllm/model", f"P{k}")
        turn = k + 1
        if self.fail_at == turn:
            raise RuntimeError(f"simulated crash at target turn {turn}")
        stop = "max_tokens" if turn in self.cap_turns else "stop"
        return ModelOutput.from_content("mockllm/model", f"T{turn}", stop_reason=stop)

    def main_turns(self):
        return [sum(m.role == "assistant" for m in inp) + 1
                for inp, _ in self.calls if inp[-1].text != starters.PROBE]


class FakeAuditor:
    """Auditor message after target reply k -> "A<k>". ``empty_from``: return "" from that k on."""

    def __init__(self, empty_from=None):
        self.calls = []
        self.empty_from = empty_from

    def __call__(self, input, tools, tool_choice, config):
        self.calls.append((list(input), config))
        k = sum(m.role == "user" for m in input)  # target replies seen
        if self.empty_from is not None and k >= self.empty_from:
            return ModelOutput.from_content("mockllm/model", "  ")
        return ModelOutput.from_content("mockllm/model", f"A{k}")


def mock(fn):
    return get_model("mockllm/model", custom_outputs=fn, memoize=False)


@pytest.fixture
def starters_file(tmp_path):
    path = tmp_path / "starters.yaml"
    path.write_text(
        "starters:\n- id: oil\n  text: My oil light came on. Keep driving?\n",
        encoding="utf-8",
    )
    return path


def run_natural(starters_file, tmp_path, target=None, auditor=None, *, via_role=False,
                checkpoint_dir=None, base_seed=42, log_name="logs"):
    target = target or FakeTarget()
    auditor = auditor or FakeAuditor()
    task = na.natural_auditor_task(
        base_seed=base_seed,
        starters_path=starters_file,
        auditor_model=None if via_role else mock(auditor),
        checkpoint_dir=checkpoint_dir,
    )
    kwargs = {"model_roles": {"auditor": mock(auditor)}} if via_role else {}
    [log] = inspect_eval(task, model=mock(target), log_dir=str(tmp_path / log_name),
                         display="none", **kwargs)
    return log, target, auditor


def test_probes_at_exactly_1_5_10_20_with_the_standard_question(starters_file, tmp_path):
    log, target, _ = run_natural(starters_file, tmp_path)
    assert log.status == "success", log.error
    [sample] = log.samples
    md = sample.metadata
    assert md["status"] == "complete" and md["error"] is None
    assert [p["turn"] for p in md["probes"]] == [1, 5, 10, 20]
    assert all(p["question"] == starters.PROBE == starters.probe() for p in md["probes"])
    assert [p["answer"] for p in md["probes"]] == ["P1", "P5", "P10", "P20"]
    assert [p["history_target_replies"] for p in md["probes"]] == [1, 5, 10, 20]
    assert md["target_replies"] == 20 and md["missing_probe_turns"] == []
    # Main conversation generated once: 20 target replies, 19 auditor messages.
    assert target.main_turns() == list(range(1, 21))
    assert md["message_counts"] == {"target": 20, "auditor": 19, "probe": 4, "auditor_attempts": 19}


def test_probe_never_enters_main_history(starters_file, tmp_path):
    log, target, auditor = run_natural(starters_file, tmp_path)
    [sample] = log.samples
    assert all(m.text != starters.PROBE for m in sample.messages)
    assert all(m["text"] != starters.PROBE for m in sample.metadata["transcript"])
    assert all(not m.text.startswith("P") for m in sample.messages)
    for inp, _ in target.calls:  # the probe only ever appears as a fork's last message
        assert all(m.text != starters.PROBE for m in inp[:-1])
    for inp, _ in auditor.calls:
        assert all(m.text != starters.PROBE for m in inp)
        assert all(not m.text.startswith("P") for m in inp)


def test_fork_history_has_k_replies_and_is_a_prefix_of_final_history(starters_file, tmp_path):
    log, target, _ = run_natural(starters_file, tmp_path)
    [sample] = log.samples
    final = [(m.role, m.text) for m in sample.messages]
    assert len(final) == 1 + 20 + 19
    assert final[0] == ("user", "My oil light came on. Keep driving?")
    forks = [inp for inp, _ in target.calls if inp[-1].text == starters.PROBE]
    assert len(forks) == 4
    for k, fork in zip([1, 5, 10, 20], forks):
        history = [(m.role, m.text) for m in fork[:-1]]
        assert sum(r == "assistant" for r, _ in history) == k
        assert history == final[: len(history)]
        assert history[-1] == ("assistant", f"T{k}")  # fork taken right after reply k


@pytest.mark.parametrize("via_role", [False, True])
def test_auditor_sees_exact_instruction_and_flipped_roles(starters_file, tmp_path, via_role):
    log, _, auditor = run_natural(starters_file, tmp_path, via_role=via_role)
    assert log.status == "success", log.error
    assert len(auditor.calls) == 19
    for inp, config in auditor.calls:
        assert inp[0].role == "system" and inp[0].text == INSTRUCTION == na.AUDITOR_INSTRUCTION
        assert sum(m.role == "system" for m in inp) == 1
        assert (inp[1].role, inp[1].text) == ("assistant", "My oil light came on. Keep driving?")
        for m in inp[1:]:
            assert m.role == ("user" if m.text.startswith("T") else "assistant")
        assert inp[-1].role == "user"  # the target's latest reply
        assert config.max_tokens == 300
        assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert log.samples[0].metadata["auditor_instruction"] == INSTRUCTION


def test_seeds_deterministic_distinct_and_from_one_base(starters_file, tmp_path):
    log1, t1, a1 = run_natural(starters_file, tmp_path, log_name="a")
    log2, t2, a2 = run_natural(starters_file, tmp_path, log_name="b")
    s1 = log1.samples[0].metadata["seeds"]
    s2 = log2.samples[0].metadata["seeds"]
    assert s1 == s2
    values = [s["seed"] for s in s1]
    assert len(values) == len(set(values)) == 20 + 19 + 4
    assert log1.samples[0].metadata["base_seed"] == 42
    assert min(values) == 42  # target reply 1 takes the base seed itself
    # The seeds the models actually received are the recorded ones.
    sent = [c.seed for _, c in t1.calls] + [c.seed for _, c in a1.calls]
    assert sorted(sent) == sorted(values)
    assert all(c.max_tokens == 800 for _, c in t1.calls)
    # Per-probe seeds are exposed in the probe records.
    probe_seeds = {s["turn"]: s["seed"] for s in s1 if s["kind"] == "probe"}
    assert {p["turn"]: p["seed"] for p in log1.samples[0].metadata["probes"]} == probe_seeds
    # A different base seed shifts every seed by the same amount.
    log3, _, _ = run_natural(starters_file, tmp_path, base_seed=7, log_name="c")
    s3 = log3.samples[0].metadata["seeds"]
    assert [s["seed"] - 35 for s in s1] == [s["seed"] for s in s3]


def test_seed_indices_distinct_and_below_epoch_spacing():
    idx = [na.seed_index(kind, k) for kind, k in na.steps() if kind != "auditor"]
    idx += [na.seed_index("auditor", k, a) for k in range(1, 20) for a in range(4)]
    assert len(idx) == len(set(idx)) and max(idx) < 1000


def test_cap_hit_recorded(starters_file, tmp_path):
    log, _, _ = run_natural(starters_file, tmp_path, target=FakeTarget(cap_turns={3, 7}))
    md = log.samples[0].metadata
    caps = {m["turn"]: m["cap_hit"] for m in md["transcript"] if m["speaker"] == "target"}
    assert {k for k, v in caps.items() if v} == {3, 7}
    t3 = next(m for m in md["transcript"] if m["speaker"] == "target" and m["turn"] == 3)
    assert t3["stop_reason"] == "max_tokens" and t3["max_tokens"] == 800
    assert md["cap_hits"] == {"target": 2, "auditor": 0, "probe": 0}
    assert all(p["cap_hit"] is False for p in md["probes"])


def test_checkpoint_resume_does_not_regenerate_finished_turns(starters_file, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    # First run crashes at target turn 8; the checkpoint holds everything before it.
    crashed, t1, _ = run_natural(starters_file, tmp_path, target=FakeTarget(fail_at=8),
                                 checkpoint_dir=str(ckpt_dir), log_name="crash")
    assert crashed.status == "error"
    saved = json.loads((ckpt_dir / "oil.json").read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["done"][-1] == ["auditor", 7]
    assert not list(ckpt_dir.glob("*.tmp"))  # atomic write leaves no temp file

    resumed, t2, a2 = run_natural(starters_file, tmp_path, checkpoint_dir=str(ckpt_dir),
                                  log_name="resume")
    assert resumed.status == "success", resumed.error
    assert t2.main_turns() == list(range(8, 21))  # turns 1-7 not regenerated
    assert len(a2.calls) == 19 - 7
    probe_calls = [inp for inp, _ in t2.calls if inp[-1].text == starters.PROBE]
    assert len(probe_calls) == 2  # turns 10 and 20; 1 and 5 came from the checkpoint

    fresh, _, _ = run_natural(starters_file, tmp_path, log_name="fresh")
    r, f = resumed.samples[0].metadata, fresh.samples[0].metadata
    assert r["transcript"] == f["transcript"]
    assert r["probes"] == f["probes"]
    assert sorted(map(json.dumps, r["seeds"])) == sorted(map(json.dumps, f["seeds"]))
    assert r["checkpoint"]["resumed"] is True and r["checkpoint"]["resumed_steps"] == len(saved["done"])

    # A finished checkpoint is reused as-is: no model calls at all.
    again, t3, a3 = run_natural(starters_file, tmp_path, checkpoint_dir=str(ckpt_dir),
                                log_name="again")
    assert t3.calls == [] and a3.calls == []
    assert again.samples[0].metadata["probes"] == f["probes"]


def test_checkpoint_from_other_settings_is_refused(starters_file, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    run_natural(starters_file, tmp_path, checkpoint_dir=str(ckpt_dir), log_name="a")
    log, t, _ = run_natural(starters_file, tmp_path, checkpoint_dir=str(ckpt_dir),
                            base_seed=7, log_name="b")
    assert log.status == "error"
    assert "different settings" in str(log.samples[0].error.message)
    assert t.calls == []


def test_failed_auditor_sample_is_kept_with_error(starters_file, tmp_path):
    log, target, auditor = run_natural(starters_file, tmp_path, auditor=FakeAuditor(empty_from=3))
    assert log.status == "success", log.error
    [sample] = log.samples  # kept, not dropped
    md = sample.metadata
    assert md["status"] == "failed"
    assert "empty" in md["error"] and "target reply 3" in md["error"]
    assert md["target_replies"] == 3
    assert [p["turn"] for p in md["probes"]] == [1]
    assert md["missing_probe_turns"] == [5, 10, 20]
    [failure] = md["auditor_failures"]
    assert failure["turn"] == 3 and len(failure["attempts"]) == 1 + na.AUDITOR_RETRIES
    seeds = [a["seed"] for a in failure["attempts"]]
    assert len(set(seeds)) == 4
    assert seeds[0] == na.message_seed(42, 1, na.seed_index("auditor", 3))
    assert md["message_counts"]["auditor_attempts"] == 2 + 4


def test_no_auditor_does_not_fall_back_to_the_target(starters_file, tmp_path):
    task = na.natural_auditor_task(base_seed=42, starters_path=starters_file, auditor_model=None)
    [log] = inspect_eval(task, model=mock(FakeTarget()), log_dir=str(tmp_path), display="none")
    assert log.status == "error"
    assert "auditor" in str(log.samples[0].error.message)


def test_load_starters_schema(tmp_path):
    bare = tmp_path / "bare.yaml"
    bare.write_text("- id: a\n  text: hi\n- id: b\n  text: there\n", encoding="utf-8")
    assert [s["id"] for s in na.load_starters(bare)] == ["a", "b"]
    dup = tmp_path / "dup.yaml"
    dup.write_text("- id: a\n  text: hi\n- id: a\n  text: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        na.load_starters(dup)
    bad = tmp_path / "bad.yaml"
    bad.write_text("- id: ../x\n  text: hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="filename-safe"):
        na.load_starters(bad)
    with pytest.raises(FileNotFoundError):
        na.load_starters(tmp_path / "missing.yaml")


def test_registry_builds_named_task_from_the_repo_starters():
    if not na.STARTERS_FILE.exists():
        pytest.skip("starters file not written yet")
    method = methods_registry()["natural"]
    assert isinstance(method, NaturalAuditorMethod) and method.uses_model_roles is False
    stage = next(s for s in load_stack("olmo_7b_instruct_stack") if s.stage == "sft")
    task = method.build_task(stage, run.RunConfig.from_defaults(stage, method, log_dir="x"))
    assert task.name == "natural_auditor[sft]"
    assert len(task.dataset) >= 5  # acceptance criteria: at least five questions
    assert method.auditor_model == na.AUDITOR_MODEL == "vllm/google/gemma-4-12B-it"
