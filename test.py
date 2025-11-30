import torch
from module.modeling_switcher import SwitcherConfig, SwitcherModelForCausalLM
from transformers import AutoTokenizer
from torchinfo import summary

from fla.models import GatedDeltaNetForCausalLM

config = SwitcherConfig()
model = SwitcherModelForCausalLM(config).cuda()
model.eval()
# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b", legacy=False)
tokenizer.pad_token = tokenizer.eos_token

print(model)
# print(sum(p.numel() for p in model.parameters()) / 1_000_000, 'M')

sentences = ['i am in the center of ', 'please give me an explanation about Fourier transformation in simple words']
encodings = tokenizer(sentences, return_tensors='pt', padding=True)

summary(model)

# with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
#  output = model(encodings.input_ids.cuda(), attention_mask=encodings.attention_mask.cuda(), use_cache=False)
# print(output)


from fla.models import DeltaNetForCausalLM
model = DeltaNetForCausalLM.from_pretrained('m-a-p/340M-20B-DeltaNet-pure', torch_dtype=torch.bfloat16, local_files_only=True).cuda()
tokenizer = AutoTokenizer.from_pretrained('m-a-p/340M-20B-DeltaNet-pure', local_files_only=True)
print(tokenizer.vocab_size)
print(model)
summary(model)
# print(sum(p.numel() for p in model.parameters()) / 1_000_000, 'M')
# 
