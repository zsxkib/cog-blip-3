import os
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)
import torch
import requests
from PIL import Image

MODEL_CACHE = "models"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


# define the prompt template
def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


# load models
model_name_or_path = "Salesforce/blip3-phi3-mini-instruct-r-v1"
model = AutoModelForVision2Seq.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    force_download=False,
    cache_dir=MODEL_CACHE,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
    legacy=False,
    force_download=False,
    cache_dir=MODEL_CACHE,
)
image_processor = AutoImageProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    force_download=False,
    cache_dir=MODEL_CACHE,
)
tokenizer = model.update_special_tokens(tokenizer)

# craft a test sample
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
query = "how many dogs are in the picture?"

model = model.cuda()
inputs = image_processor([raw_image], return_tensors="pt", image_aspect_ratio="anyres")
prompt = apply_prompt_template(query)
language_inputs = tokenizer([prompt], return_tensors="pt")
inputs.update(language_inputs)
inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
generated_text = model.generate(
    **inputs,
    image_size=[raw_image.size],
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_new_tokens=768,
    top_p=None,
    num_beams=1,
    stopping_criteria=[EosListStoppingCriteria()],
)
prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split(
    "<|end|>"
)[0]
print("==> prediction: ", prediction)
# output: ==> prediction: There is one dog in the picture.
