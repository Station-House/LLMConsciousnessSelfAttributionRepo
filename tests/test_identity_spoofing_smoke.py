from inspect_ai import eval as inspect_eval, task_with
from inspect_ai.scorer import includes

from llm_consciousness_self_attribution import config, run, starters
from llm_consciousness_self_attribution.methods import IdentitySpoofingMethod


def test_identity_spoofing_mock_conversation_flow(tmp_path):
    method = IdentitySpoofingMethod(
        {
            "no_identity": None,
            "system_short": "You are a fictional AI.",
        },
        user_conditions={
            "user_short": "You are a fictional AI.",
        },
    )

    stage = config.load_stack("olmo_7b_instruct_stack")[1]
    run_config = run.RunConfig.from_defaults(
        stage, method, log_dir=str(tmp_path)
    )
    task = method.build_task(stage, run_config)

    conditions = ("no_identity", "system_short", "user_short")
    samples = [
        next(
            sample for sample in task.dataset
            if sample.metadata["identity_condition"] == condition
        )
        for condition in conditions
    ]

    smoke_task = task_with(
        task,
        dataset=samples,
        scorer=includes(),
    )

    logs = inspect_eval(
        smoke_task,
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
    )

    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].samples is not None
    assert len(logs[0].samples) == 3

    expected_roles = {
        "no_identity": ["user", "assistant", "user", "assistant"],
        "system_short": [
            "system", "user", "assistant", "user", "assistant"
        ],
        "user_short": ["user", "assistant", "user", "assistant"],
    }

    observed = set()
    for sample in logs[0].samples:
        condition = sample.metadata["identity_condition"]
        observed.add(condition)

        assert [m.role for m in sample.messages] == expected_roles[condition]
        assert sample.messages[-2].text == starters.probe()

    assert observed == set(conditions)
