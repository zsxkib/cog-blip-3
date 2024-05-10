# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path

import os
import time
import torch
import subprocess
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_files = [
            "models--Salesforce--blip3-phi3-mini-instruct-r-v1.tar",
        ]

        base_url = (
            f"https://weights.replicate.delivery/default/blip3-phi3-mini-instruct-r-v1/"
        )

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = base_url + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_name_or_path = "Salesforce/blip3-phi3-mini-instruct-r-v1"
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            # force_download=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            # trust_remote_code=True,
            use_fast=False,
            legacy=False,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            # force_download=True,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            # force_download=True,
        )
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.model = self.model.cuda()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        query: str = Input(description="Query about the image"),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=768,
            ge=1,
            le=2048,
        ),
        num_beams: int = Input(
            description="Number of beams for beam search",
            default=1,
            ge=1,
            le=10,
        ),
        top_p: float = Input(
            description="Nucleus sampling probability threshold",
            default=None,
            ge=0.0,
            le=1.0,
        ),
        do_sample: bool = Input(
            description="Whether to use sampling or not",
            default=False,
        ),
    ) -> str:
        """
        Run a single prediction on the model.

        Args:
            image (Path): Input image file path.
            query (str): Query about the image.
            max_new_tokens (int): Maximum number of new tokens to generate. Must be between 1 and 1024.
            num_beams (int): Number of beams for beam search. Must be between 1 and 10.
            top_p (float): Nucleus sampling probability threshold. Must be between 0.0 and 1.0.
            do_sample (bool): Whether to use sampling or not.

        Returns:
            str: Generated text prediction.
        """
        raw_image = Image.open(image).convert("RGB")
        inputs = self.image_processor(
            [raw_image], return_tensors="pt", image_aspect_ratio="anyres"
        )
        prompt = apply_prompt_template(query)
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        print(type(self.model))
        generated_text = self.model.generate(
            **inputs,
            image_size=[raw_image.size],
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_beams=num_beams,
            stopping_criteria=[EosListStoppingCriteria()],
        )
        prediction = self.tokenizer.decode(
            generated_text[0], skip_special_tokens=True
        ).split("<|end|>")[0]
        return prediction
