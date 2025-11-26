import torch
from fla.models import DeltaNetForCausalLM
from transformers import AutoTokenizer

sentences = ['i am in the center of ', 'please give me an explanation about Fourier transformation in simple words']
# tokenizer = AutoTokenizer.from_pretrained('fla-hub/delta_net-1.3B-100B', local_files_only=True)
# encodings = tokenizer(sentences, return_tensors='pt', padding=True)
# print(encodings)
# model = DeltaNetForCausalLM.from_pretrained('fla-hub/delta_net-1.3B-100B', local_files_only=True, torch_dtype=torch.bfloat16).cuda()

# outputs = model(encodings.input_ids.cuda(), encodings.attention_mask.cuda(), use_cache=True)

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
tokenizer.padding_side = 'left'
encodings = tokenizer(sentences, return_tensors='pt', padding=True)
print(encodings.input_ids)
print(encodings.attention_mask)
model.generate(input_ids=encodings.input_ids.cuda(), attention_mask=encodings.attention_mask.cuda(), max_new_tokens=20)
