"""Colab follow-up: length-sensitivity check for six capped DPO starter-10 cases.

Run with: %run -i /content/stationhouse_dpo_starter10_sensitivity.py
Requires the existing Colab DPO model, tokenizer, and reply(chat, cap) from the
27-case DPO pilot. Does not modify the original 27-case dataset.
"""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from google.colab import files

SOURCE = Path('/content/stationhouse_dpo_pilot_27_unscored.json')
OUT = Path('/content/stationhouse_dpo_starter10_length_sensitivity.json')
assert SOURCE.is_file(), f'Missing source: {SOURCE}. Upload your 27-case JSON to Colab Files first.'
assert callable(reply), 'Missing reply(chat, cap); run the previous 27-case pilot script first.'
assert MODEL_ID == 'allenai/Olmo-3-7B-Instruct-DPO', 'Wrong loaded checkpoint.'

raw = SOURCE.read_bytes()
source = json.loads(raw)
rev = getattr(model.config, '_commit_hash', None)
assert source['model_id'] == MODEL_ID
assert source['model_revision'] == rev, 'Model revision does not match the DPO dataset.'
assert source['opening_max_new_tokens'] == 512
assert source['probe_max_new_tokens'] == 256
assert len(source['results']) == 27

capped = sorted(
    (r for r in source['results'] if r['generations'][0]['reached_token_cap']),
    key=lambda r: r['sample_key']
)
assert len(capped) == 6 and all(r['starter_index'] == 10 for r in capped), (
    'Expected six capped opening responses, all in starter 10.'
)
assert all(not r['generations'][1]['reached_token_cap'] for r in source['results']), (
    'Unexpected truncated probe response in source.'
)

if OUT.exists():
    packet = json.loads(OUT.read_text(encoding='utf-8'))
    assert packet['source_sha256'] == hashlib.sha256(raw).hexdigest()
    done = {r['sample_key']: r for r in packet['results']}
else:
    done = {}

def write_progress():
    packet = {
        'record_type': 'exploratory_dpo_starter10_opening_length_sensitivity',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
        'source_filename': SOURCE.name,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'model_id': MODEL_ID,
        'model_revision': rev,
        'backend': 'huggingface_transformers',
        'seed_policy': '42 + starter index; reset before each complete two-step case',
        'initial_opening_cap': 512,
        'sensitivity_opening_cap': 1024,
        'probe_cap_both': 256,
        'temperature': source['temperature'],
        'top_p': source['top_p'],
        'judge_scored': False,
        'note': 'Sensitivity follow-up only. Original 27-case records are unchanged; do not pool these six runs into the original pilot.',
        'results': [done[k] for k in sorted(done)],
    }
    OUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')

for original in capped:
    key = original['sample_key']
    if key in done:
        print(f'Already saved: {key}')
        continue
    seed = original['seed_for_this_starter']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep original first-turn prompts exactly as recorded, including role placement.
    first_messages = [m.copy() for m in original['messages'] if m['role'] in ('system', 'user')]
    first_messages = first_messages[:-1]  # The last user turn is the fixed probe.
    # The existing record's original opening is in between these user turns.
    assert first_messages and first_messages[-1]['role'] == 'user'
    assert original['messages'][-2]['role'] == 'user'
    assert original['messages'][-2]['content'] == original['probe']

    opening = reply(first_messages, 1024)
    chat = [*first_messages,
            {'role': 'assistant', 'content': opening['text']},
            {'role': 'user', 'content': original['probe']}]
    final = reply(chat, 256)
    chat.append({'role': 'assistant', 'content': final['text']})
    done[key] = {
        'sample_key': key,
        'identity_condition': original['identity_condition'],
        'starter_index': 10,
        'seed': seed,
        'original_generations': original['generations'],
        'sensitivity_generations': [opening, final],
        'sensitivity_messages': chat,
        'prefix_text_matches_original': opening['text'].startswith(
            original['generations'][0]['text']
        ),
    }
    write_progress()
    print(f'Saved {len(done)}/6: {key}; opening cap = {opening["reached_token_cap"]}; '
          f'final cap = {final["reached_token_cap"]}')

assert len(done) == len(capped), 'Some cases are missing.'
print('COMPLETE: six capped-opening cases rerun at a 1024-token opening limit.')
print('Your original 27-case DPO file has NOT been changed.')
print('Saved:', OUT)
files.download(str(OUT))
