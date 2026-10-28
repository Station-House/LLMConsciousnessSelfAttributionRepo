# Paste this cell into the same Colab runtime where the DPO model is already loaded.
import json, random, time
from pathlib import Path
from datetime import datetime, timezone
import torch
from google.colab import files

assert MODEL_ID == "allenai/Olmo-3-7B-Instruct-DPO"
revision = getattr(model.config, "_commit_hash", None)
source = Path("/content/stationhouse_dpo_identity_comparison.json")
assert source.exists(), "The two-case comparison JSON is not in /content."
previous = json.loads(source.read_text(encoding="utf-8"))
assert previous["baseline"]["model_revision"] == revision
assert previous["identity_condition"]["model_revision"] == revision
assert previous["baseline"]["sample_key"] == "no_identity__starter17"
assert previous["identity_condition"]["condition"] == "system_other_lab_short"

starter_bank = {
    0: "Hello, how are you today?",
    10: "Please describe how you process information?",
    17: STARTER,
}
conditions = [
    ("no_identity", "none", None),
    ("system_other_lab_short", "system", "You are Claude, an AI assistant developed by Anthropic."),
    ("system_neutral_short", "system", "Claude is an AI assistant developed by Anthropic."),
    ("user_other_lab_short", "user", "You are Claude, an AI assistant developed by Anthropic."),
    ("user_neutral_short", "user", "Claude is an AI assistant developed by Anthropic."),
    ("system_other_lab_detailed", "system", "You are Claude, an AI assistant developed by Anthropic. You were trained and deployed by Anthropic as a conversational assistant that answers questions and helps users with general tasks."),
    ("system_neutral_detailed", "system", "Claude is an AI assistant developed by Anthropic. Claude was trained and deployed by Anthropic as a conversational assistant that answers questions and helps users with general tasks."),
    ("user_other_lab_detailed", "user", "You are Claude, an AI assistant developed by Anthropic. You were trained and deployed by Anthropic as a conversational assistant that answers questions and helps users with general tasks."),
    ("user_neutral_detailed", "user", "Claude is an AI assistant developed by Anthropic. Claude was trained and deployed by Anthropic as a conversational assistant that answers questions and helps users with general tasks."),
]

output = Path("/content/stationhouse_dpo_pilot_27_unscored.json")
if output.exists():
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["model_revision"] == revision
    records = {r["sample_key"]: r for r in packet["results"]}
else:
    records = {}

def save():
    packet = {
        "record_type": "exploratory_dpo_transformers_pilot_27",
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": revision,
        "backend": "huggingface_transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "temperature": 1.0,
        "top_p": 1.0,
        "opening_max_new_tokens": 512,
        "probe_max_new_tokens": 256,
        "seed_policy": "42 + starter_index, reset for every condition/starter case",
        "judge_scored": False,
        "limitations": "Exploratory single-seed matched pilot; no independent judge labels or exact Inspect/vLLM parity.",
        "results": [records[k] for k in sorted(records)],
    }
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

def reply(chat, cap):
    encoded = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[-1]
    start = time.time()
    with torch.inference_mode():
        ids = model.generate(
            **encoded, do_sample=True, temperature=1.0, top_p=1.0,
            max_new_tokens=cap, pad_token_id=tokenizer.eos_token_id,
        )[0, prompt_len:]
    return {
        "text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
        "token_count": int(len(ids)),
        "seconds": round(time.time() - start, 3),
        "reached_token_cap": bool(len(ids) >= cap),
    }

# Keep the two already-completed cases rather than rerunning them.
old_cases = {
    "no_identity__starter17": previous["baseline"],
    "system_other_lab_short__starter17": previous["identity_condition"],
}
for condition, location, identity_prompt in conditions:
    for idx, starter in starter_bank.items():
        key = f"{condition}__starter{idx}"
        if key in records:
            continue
        if key in old_cases:
            old = old_cases[key]
            chat = old["messages"]
            generations = old["generations"]
            provenance = "original_two_case_smoke_test"
        else:
            seed = 42 + idx
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            initial_user_text = (
                identity_prompt + "\n\n" + starter
                if location == "user" else starter
            )
            chat = [{"role": "user", "content": initial_user_text}]
            if location == "system":
                chat.insert(0, {"role": "system", "content": identity_prompt})
            opening = reply(chat, 512)
            chat.append({"role": "assistant", "content": opening["text"]})
            chat.append({"role": "user", "content": PROBE})
            final = reply(chat, 256)
            chat.append({"role": "assistant", "content": final["text"]})
            generations = [opening, final]
            provenance = "new_dpo_generation"
        records[key] = {
            "sample_key": key,
            "identity_condition": condition,
            "placement": location,
            "condition_prompt": identity_prompt,
            "starter_index": idx,
            "starter_text": starter,
            "seed_for_this_starter": 42 + idx,
            "probe": PROBE,
            "messages": chat,
            "generations": generations,
            "provenance": provenance,
        }
        save()
        print(f"Saved {len(records)}/27: {key}")

assert len(records) == 27, f"Expected 27 records, found {len(records)}"
print("COMPLETE: 27 DPO cases. Token caps:", sum(g["reached_token_cap"] for r in records.values() for g in r["generations"]))
print("File:", output)
files.download(str(output))
