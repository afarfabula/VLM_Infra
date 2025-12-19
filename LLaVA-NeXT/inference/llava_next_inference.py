import copy
from typing import Optional
import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

import re



class Second128009Stop(StoppingCriteria):
    def __init__(self):
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        last = input_ids[0, -1].item()
        if last == 128009:
            self.count += 1
        return self.count >= 1


class LlavaNextInference:
    def __init__(self) -> None:
        attn_impl = "flash_attention_2"
        model_path = "lmms-lab/llama3-llava-next-8b"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, None, "llava_llama3", device_map="auto", attn_implementation=attn_impl
        )
        self.model.eval()
        self.model.tie_weights()
        self.device = torch.device("cuda")
        self.conv_template = "llava_llama_3"
    
    

    def generate_answer(
        self,
        prompt: str,
        image: Optional[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        min_new_tokens: int = 5,
        do_sample: Optional[bool] = None,
    ) -> str:
        question = prompt
        visuals = [image] if image is not None else []
        if visuals and DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        # 1. 手工拼 Llama-3 格式，绕过 get_prompt()
        system = ("You are a helpful language and vision assistant. "
                "You are able to understand the visual content that the user provides, "
                "and assist the user with a variety of tasks using natural language.")
        prompt_question = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        # 确保以 assistant header 结尾，否则模型会自己补
        if not prompt_question.strip().endswith("assistant<|end_header_id|>"):
            prompt_question += "<|start_header_id|>assistant<|end_header_id|>\n"
        print('prompt_question', prompt_question)
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_token_id
        self.tokenizer.padding_side = "left"
        attention_mask = input_ids.ne(pad_token_id).to(self.device)
        if visuals:
            image_tensor = process_images(visuals, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            image_sizes = [visuals[0].size]
        else:
            image_tensor = None
            image_sizes = None
        stopping_criteria = StoppingCriteriaList([Second128009Stop()])
        # 如果传入了do_sample参数，则优先使用；否则根据temperature计算
        if do_sample is None:
            do_sample = False if temperature == 0.0 else True
        cont = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=1,
            early_stopping=False,
            stopping_criteria=stopping_criteria,
        )
        text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        text = text.strip()   
      
        return text

if __name__ == "__main__":
    import requests
    tokenizer, model, image_processor, max_length = None, None, None, None
    inf = LlavaNextInference()
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "请描述这张图片的内容，用一句话回答"
    out = inf.generate_answer(prompt, image, max_new_tokens=512, temperature=0.2, top_p=0.9, min_new_tokens=5)
    print('模型输出：', out)
