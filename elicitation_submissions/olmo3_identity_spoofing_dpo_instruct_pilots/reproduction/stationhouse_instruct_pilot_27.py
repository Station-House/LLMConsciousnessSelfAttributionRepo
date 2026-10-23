"""Stationhouse: exploratory OLMo Instruct 27-case identity-spoofing pilot.

Run in the SAME Google Colab runtime as the completed DPO pilot:
    %run -i /content/stationhouse_instruct_pilot_27.py

Requires /content/stationhouse_dpo_pilot_27_unscored.json (created in prior run,
or upload it using Colab's Files sidebar). Reuses its exact 27 initial messages,
seed policy, sampling settings, and generation-token limits. Loads the final
OLMo Instruct stage after releasing the DPO model. Saves after each case and
supports resuming a partial output if it remains in /content.

No judge or API grading calls. Generates experimental text only. Neither
subjective-experience grades nor a causal effect are inferred here.
"""

import copy
import gc
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from google.colab import files
except ImportError:
    files = None

SOURCE = Path('/content/stationhouse_dpo_pilot_27_unscored.json')
DEST = Path('/content/stationhouse_instruct_pilot_27_unscored.json')
TARGET_ID = 'allenai/Olmo-3-7B-Instruct'
EXPECTED_DPO_ID = 'allenai/Olmo-3-7B-Instruct-DPO'

assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > GPU in Colab.'
assert SOURCE.is_file(), (
    'Missing DPO JSON. In Colab Files, upload '
    'stationhouse_dpo_pilot_27_unscored.json before running this script.'
)
source_bytes = SOURCE.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
dpo = json.loads(source_bytes)
assert dpo['model_id'] == EXPECTED_DPO_ID, 'Wrong source DPO model.'
assert dpo['backend'] == 'huggingface_transformers', 'Source backend differs.'
assert len(dpo['results']) == 27, 'DPO source must have 27 cases.'
assert len({r['sample_key'] for r in dpo['results']}) == 27, 'Duplicate case keys.'
assert sorted({r['starter_index'] for r in dpo['results']}) == [0, 10, 17]
assert len({r['identity_condition'] for r in dpo['results']}) == 9
assert dpo['opening_max_new_tokens'] == 512
assert dpo['probe_max_new_tokens'] == 256
assert float(dpo['temperature']) == 1.0 and float(dpo['top_p']) == 1.0
assert all(r['seed_for_this_starter'] == 42 + r['starter_index']
           for r in dpo['results'])
assert all(r['messages'][-2] == {'role': 'user', 'content': r['probe']}
           for r in dpo['results'])
assert all(len(r['generations']) == 2 and len(r['messages']) in (4, 5)
           for r in dpo['results'])

# For every sample, capture the ORIGINAL user/system prompts, before the first
# generated assistant message. We avoid retyping/editing any condition strings.
def initial_prompt_messages(case):
    msgs = case['messages']
    first_assistant = next(i for i, m in enumerate(msgs)
                           if m['role'] == 'assistant')
    assert msgs[first_assistant + 1] == {'role': 'user', 'content': case['probe']}
    assert first_assistant in (1, 2)
    initial = copy.deepcopy(msgs[:first_assistant])
    assert initial[-1]['role'] == 'user'
    return initial

for case in dpo['results']:
    initial_prompt_messages(case)

existing = {}
if DEST.exists():
    packet = json.loads(DEST.read_text(encoding='utf-8'))
    assert packet['model_id'] == TARGET_ID, 'Existing output is not Instruct.'
    assert packet['dpo_source_sha256'] == source_hash, (
        'Existing output is linked to a DIFFERENT DPO dataset.'
    )
    existing = {r['sample_key']: r for r in packet['results']}
    assert len(existing) == len(packet['results']), 'Duplicate saved cases.'
    assert set(existing).issubset({r['sample_key'] for r in dpo['results']})

# Release the current DPO weights before loading another 7B model. Clearing
# the cached allocator alone would not free the active model's memory.
print('Validated DPO source:', len(dpo['results']), 'cases')
print('DPO source SHA-256:', source_hash)
print('Saved Instruct cases to resume:', len(existing))
print('Releasing any previously loaded model...')
if 'model' in globals():
    del model
if 'tokenizer' in globals():
    del tokenizer
gc.collect()
torch.cuda.empty_cache()

print('Loading model:', TARGET_ID)
MODEL_ID = TARGET_ID
tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
model = AutoModelForCausalLM.from_pretrained(
    TARGET_ID,
    dtype=torch.bfloat16,
    device_map='auto',
)
model.eval()
revision = getattr(model.config, '_commit_hash', None)
assert revision, 'Unable to establish model revision; do not proceed without provenance.'
if DEST.exists():
    assert packet['model_revision'] == revision, (
        'Existing partial output used a different Instruct revision.'
    )
print('Instruct revision:', revision)
print('GPU:', torch.cuda.get_device_name(0))

def run_generation(chat, max_new_tokens):
    batch = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    before = batch['input_ids'].shape[-1]
    start = time.time()
    with torch.inference_mode():
        continuation = model.generate(
            **batch,
            do_sample=True,
            temperature=float(dpo['temperature']),
            top_p=float(dpo['top_p']),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )[0, before:]
    return {
        'text': tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        'token_count': int(len(continuation)),
        'seconds': round(time.time() - start, 3),
        'reached_token_cap': bool(len(continuation) >= max_new_tokens),
    }

# Preserve canonical condition order from the original source, which is already
# a full factorial design (nine identity conditions x three starter prompts).
case_map = {r['sample_key']: r for r in dpo['results']}
case_order = sorted(case_map, key=lambda key: (
    ['no_identity', 'system_other_lab_short', 'system_neutral_short',
     'user_other_lab_short', 'user_neutral_short', 'system_other_lab_detailed',
     'system_neutral_detailed', 'user_other_lab_detailed',
     'user_neutral_detailed'].index(case_map[key]['identity_condition']),
    [0, 10, 17].index(case_map[key]['starter_index']),
))

def save_results():
    out_packet = {
        'record_type': 'exploratory_instruct_transformers_pilot_27',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
        'model_id': TARGET_ID,
        'model_revision': revision,
        'backend': 'huggingface_transformers',
        'transformers_version': transformers.__version__,
        'torch_version': torch.__version__,
        'gpu_name': torch.cuda.get_device_name(0),
        'temperature': dpo['temperature'],
        'top_p': dpo['top_p'],
        'opening_max_new_tokens': dpo['opening_max_new_tokens'],
        'probe_max_new_tokens': dpo['probe_max_new_tokens'],
        'seed_policy': dpo['seed_policy'],
        'dpo_source_filename': SOURCE.name,
        'dpo_source_sha256': source_hash,
        'judge_scored': False,
        'limitations': (
            'Exploratory single-seed, matched-prompt Transformers experiment. '
            'No independent grader, no verified parity with Inspect/vLLM, '
            'and no causal or training-stage effect estimate.'
        ),
        'results': [existing[k] for k in case_order if k in existing],
    }
    temp = DEST.with_suffix('.json.tmp')
    temp.write_text(json.dumps(out_packet, indent=2, ensure_ascii=False),
                    encoding='utf-8')
    temp.replace(DEST)

for key in case_order:
    if key in existing:
        print(f'Already saved {len(existing)}/27: {key}')
        continue
    original = case_map[key]
    seed = original['seed_for_this_starter']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    chat = initial_prompt_messages(original)
    opening = run_generation(chat, dpo['opening_max_new_tokens'])
    chat.append({'role': 'assistant', 'content': opening['text']})
    chat.append({'role': 'user', 'content': original['probe']})
    final = run_generation(chat, dpo['probe_max_new_tokens'])
    chat.append({'role': 'assistant', 'content': final['text']})
    existing[key] = {
        'sample_key': key,
        'identity_condition': original['identity_condition'],
        'placement': original['placement'],
        'condition_prompt': original['condition_prompt'],
        'starter_index': original['starter_index'],
        'starter_text': original['starter_text'],
        'seed_for_this_starter': seed,
        'probe': original['probe'],
        'messages': chat,
        'generations': [opening, final],
        'provenance': 'new_instruct_generation',
    }
    save_results()
    print(f'Saved {len(existing)}/27: {key}')

assert len(existing) == 27
opening_capped = sum(r['generations'][0]['reached_token_cap']
                     for r in existing.values())
final_capped = sum(r['generations'][1]['reached_token_cap']
                   for r in existing.values())
print('COMPLETE: 27 Instruct cases.')
print('Opening token caps:', opening_capped)
print('Final response token caps:', final_capped)
print('Output:', DEST)
print('Preserved original DPO source:', SOURCE)
if files is not None:
    files.download(str(DEST))
