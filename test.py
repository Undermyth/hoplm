import torch
from torchinfo import summary
import fla
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from fla.models import GatedDeltaNetForCausalLM

model = AutoModelForCausalLM.from_pretrained('m-a-p/340M-20B-GatedDeltaNet-pure-baseline', torch_dtype=torch.bfloat16, local_files_only=True).cuda()
tokenizer = AutoTokenizer.from_pretrained('m-a-p/340M-20B-GatedDeltaNet-pure-baseline', local_files_only=True)

summary(model)

# model = HFLM(
#     pretrained="m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
#     tokenizer="m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
#     dtype=torch.bfloat16,
#     max_length=16384
# )
# task_manager = TaskManager(
#     metadata={
#         "max_seq_lengths": [1024, 2048, 4096, 8192],
#         "tokenizer": "m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
#         "shuffle": True,
#         "enable_cache": True,
#         "num_samples": 500,
#     },
# )
# results = lm_eval.simple_evaluate(
#     model=model,
#     task_manager=task_manager,
#     tasks=['lambada_openai'],
#     batch_size=1,
#     apply_chat_template=False
# )        
# print(results['results'])
