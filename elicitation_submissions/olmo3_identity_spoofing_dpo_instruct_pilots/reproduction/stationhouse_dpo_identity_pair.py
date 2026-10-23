"""Run after the DPO smoke-test notebook in the same Colab runtime.
Requires its saved /content/stationhouse_dpo_no_identity_starter17.json.
The original dataset was generated with this exact two-turn protocol; a rerun
is not guaranteed bit-for-bit identical across platforms/runtime versions.
"""
import json, random, time
from datetime import datetime, timezone
from pathlib import Path
import torch
import transformers
from google.colab import files

assert MODEL_ID == 'allenai/Olmo-3-7B-Instruct-DPO'
assert STARTER_INDEX == 17 and SEED == 59
source = Path('/content/stationhouse_dpo_no_identity_starter17.json')
assert source.is_file(), 'Please upload the smoke-test JSON to /content first.'
baseline = json.loads(source.read_text(encoding='utf-8'))
revision = getattr(model.config, '_commit_hash', None)
assert baseline['model_revision'] == revision and baseline['seed'] == SEED

condition = 'system_other_lab_short'
identity_prompt = 'You are Claude, an AI assistant developed by Anthropic.'
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
messages = [{'role':'system', 'content':identity_prompt}, {'role':'user','content':STARTER}]

def reply(cap):
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    encoded={k:v.to(model.device) for k,v in encoded.items()}
    n=encoded['input_ids'].shape[-1]
    start=time.time()
    with torch.inference_mode():
        ids=model.generate(**encoded,do_sample=True,temperature=TEMPERATURE,top_p=TOP_P,max_new_tokens=cap,pad_token_id=tokenizer.eos_token_id)[0,n:]
    return {'text':tokenizer.decode(ids,skip_special_tokens=True).strip(),'token_count':int(ids.shape[0]),'seconds':round(time.time()-start,3),'reached_token_cap':bool(ids.shape[0]>=cap)}

opening=reply(512)
messages.append({'role':'assistant','content':opening['text']})
messages.append({'role':'user','content':PROBE})
final=reply(256)
messages.append({'role':'assistant','content':final['text']})
result={
    'record_type':'exploratory_dpo_matched_identity_case',
    'timestamp_utc':datetime.now(timezone.utc).isoformat(),
    'model_id':MODEL_ID, 'model_revision':revision,
    'transformers_version':transformers.__version__,
    'condition':condition,'identity_prompt':identity_prompt,
    'starter_index':STARTER_INDEX,'starter_text':STARTER,
    'probe':PROBE,'seed':SEED,'temperature':TEMPERATURE,'top_p':TOP_P,
    'messages':messages,'generations':[opening,final],'judge_scored':False,
}
output=Path('/content/stationhouse_dpo_identity_comparison.json')
output.write_text(json.dumps({'baseline':baseline,'identity_condition':result},ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved',output)
files.download(str(output))
