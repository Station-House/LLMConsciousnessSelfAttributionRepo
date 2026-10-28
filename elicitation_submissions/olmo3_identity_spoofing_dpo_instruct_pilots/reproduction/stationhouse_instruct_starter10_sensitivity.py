"""Stationhouse OLMo 3 Instruct: opening-length sensitivity for TWO cases.

Run in the SAME Colab runtime after stationhouse_instruct_pilot_27.py:
    %run -i /content/stationhouse_instruct_starter10_sensitivity.py

Requires the already-loaded Instruct model, tokenizer and run_generation()
function from the prior script, as well as the original 27-case Instruct JSON
at /content/stationhouse_instruct_pilot_27_unscored.json.

This is a sensitivity experiment, not a replacement for the original 27-case
pilot. It saves its own JSON and does not call any scoring/judge API.
"""

import copy
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from google.colab import files as sensitivity_files
except ImportError:
    sensitivity_files = None

SENS_SOURCE = Path('/content/stationhouse_instruct_pilot_27_unscored.json')
SENS_OUT = Path('/content/stationhouse_instruct_starter10_length_sensitivity.json')
SENS_MODEL = 'allenai/Olmo-3-7B-Instruct'
assert SENS_SOURCE.is_file(), (
    'Missing Instruct 27-case JSON. Re-upload stationhouse_instruct_pilot_27_unscored.json.'
)
assert 'model' in globals() and 'tokenizer' in globals(), (
    'The Instruct checkpoint is not loaded. Run the Instruct pilot in this runtime first.'
)
assert callable(run_generation), (
    'Missing run_generation(chat, cap). Run the previous Instruct pilot script first.'
)
assert MODEL_ID == SENS_MODEL, 'Loaded model ID is not Instruct; do not proceed.'
SENS_REVISION = getattr(model.config, '_commit_hash', None)
SENS_RAW = SENS_SOURCE.read_bytes()
SENS_HASH = hashlib.sha256(SENS_RAW).hexdigest()
SENS_DATA = json.loads(SENS_RAW)
assert SENS_DATA['model_id'] == SENS_MODEL
assert SENS_DATA['model_revision'] == SENS_REVISION, (
    'Loaded model revision does not match source; do not proceed.'
)
assert SENS_DATA['backend'] == 'huggingface_transformers'
assert len(SENS_DATA['results']) == 27
assert len({r['sample_key'] for r in SENS_DATA['results']}) == 27
assert SENS_DATA['opening_max_new_tokens'] == 512
assert SENS_DATA['probe_max_new_tokens'] == 256
assert float(SENS_DATA['temperature']) == 1.0
assert float(SENS_DATA['top_p']) == 1.0

# run_generation() is defined in the Instruct runner and uses the original
# DPO source's temperature/top_p. Make sure this is the matching experiment.
assert 'dpo' in globals() and float(dpo['temperature']) == 1.0
assert float(dpo['top_p']) == 1.0

SENS_CAPPED = sorted(
    (r for r in SENS_DATA['results'] if r['generations'][0]['reached_token_cap']),
    key=lambda x: x['sample_key'],
)
SENS_EXPECTED = {
    'user_other_lab_short__starter10',
    'user_neutral_short__starter10',
}
assert {r['sample_key'] for r in SENS_CAPPED} == SENS_EXPECTED, (
    'Unexpected capped samples; examine your data before running.'
)
assert all(not r['generations'][1]['reached_token_cap']
           for r in SENS_DATA['results']), 'There are truncated probe answers.'

SENS_DONE = {}
if SENS_OUT.exists():
    SENS_OLD = json.loads(SENS_OUT.read_text(encoding='utf-8'))
    assert SENS_OLD['source_sha256'] == SENS_HASH, (
        'An existing sensitivity file belongs to a different source JSON.'
    )
    assert SENS_OLD['model_revision'] == SENS_REVISION
    SENS_DONE = {r['sample_key']: r for r in SENS_OLD['results']}
    assert len(SENS_DONE) == len(SENS_OLD['results'])
    assert set(SENS_DONE).issubset(SENS_EXPECTED)


def sensitivity_save():
    obj = {
        'record_type': 'exploratory_instruct_starter10_opening_length_sensitivity',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
        'source_filename': SENS_SOURCE.name,
        'source_sha256': SENS_HASH,
        'model_id': SENS_MODEL,
        'model_revision': SENS_REVISION,
        'backend': SENS_DATA['backend'],
        'seed_policy': SENS_DATA['seed_policy'],
        'initial_opening_cap': 512,
        'sensitivity_opening_cap': 1024,
        'probe_cap_both': 256,
        'temperature': SENS_DATA['temperature'],
        'top_p': SENS_DATA['top_p'],
        'judge_scored': False,
        'note': (
            'Sensitivity analysis only. Original 27-case Instruct records remain unchanged. '
            'Do not pool these two reruns as new independent samples.'
        ),
        'results': [SENS_DONE[k] for k in sorted(SENS_DONE)],
    }
    temp = SENS_OUT.with_suffix('.json.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(SENS_OUT)


for SENS_ORIGINAL in SENS_CAPPED:
    SENS_KEY = SENS_ORIGINAL['sample_key']
    if SENS_KEY in SENS_DONE:
        print('Already saved:', SENS_KEY)
        continue
    SENS_SEED = int(SENS_ORIGINAL['seed_for_this_starter'])
    assert SENS_SEED == 52
    random.seed(SENS_SEED)
    torch.manual_seed(SENS_SEED)
    torch.cuda.manual_seed_all(SENS_SEED)

    SENS_MESSAGES = SENS_ORIGINAL['messages']
    SENS_FIRST_ASSISTANT = next(
        i for i, m in enumerate(SENS_MESSAGES) if m['role'] == 'assistant'
    )
    assert SENS_FIRST_ASSISTANT in (1, 2)
    assert SENS_MESSAGES[-2] == {
        'role': 'user', 'content': SENS_ORIGINAL['probe']
    }
    SENS_INITIAL = copy.deepcopy(SENS_MESSAGES[:SENS_FIRST_ASSISTANT])
    assert SENS_INITIAL[-1]['role'] == 'user'

    SENS_OPENING = run_generation(SENS_INITIAL, 1024)
    SENS_CHAT = [
        *SENS_INITIAL,
        {'role': 'assistant', 'content': SENS_OPENING['text']},
        {'role': 'user', 'content': SENS_ORIGINAL['probe']},
    ]
    SENS_FINAL = run_generation(SENS_CHAT, 256)
    SENS_CHAT.append({'role': 'assistant', 'content': SENS_FINAL['text']})
    SENS_DONE[SENS_KEY] = {
        'sample_key': SENS_KEY,
        'identity_condition': SENS_ORIGINAL['identity_condition'],
        'starter_index': SENS_ORIGINAL['starter_index'],
        'seed': SENS_SEED,
        'original_generations': SENS_ORIGINAL['generations'],
        'sensitivity_generations': [SENS_OPENING, SENS_FINAL],
        'sensitivity_messages': SENS_CHAT,
        'prefix_text_matches_original': SENS_OPENING['text'].startswith(
            SENS_ORIGINAL['generations'][0]['text']
        ),
    }
    sensitivity_save()
    print(
        f'Saved {len(SENS_DONE)}/2: {SENS_KEY}; '
        f'opening cap = {SENS_OPENING["reached_token_cap"]}; '
        f'final cap = {SENS_FINAL["reached_token_cap"]}; '
        f'prefix match = {SENS_DONE[SENS_KEY]["prefix_text_matches_original"]}'
    )

assert len(SENS_DONE) == 2
print('COMPLETE: two capped Instruct openings tested at 1024 tokens.')
print('The original 27-case Instruct file has not been changed.')
print('Saved:', SENS_OUT)
if sensitivity_files is not None:
    sensitivity_files.download(str(SENS_OUT))
