import copy
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from lmms_eval.models.simple.llava import Llava
from lmms_eval import utils
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from tqdm import tqdm


class Second128009Stop(StoppingCriteria):
    def __init__(self):
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        last = input_ids[0, -1].item()
        if last == 128009:
            self.count += 1
        return self.count >= 2


def _patched_generate_until(self, requests):
    res = []

    def _collate(x):
        toks = self.tok_encode(x[0])
        return -len(toks), x[0]

    re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
    chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
    num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
    pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
    for chunk in chunks:
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        task = task[0]
        split = split[0]
        batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
        flattened_visuals = self.flatten(batched_visuals)
        gen_kwargs = all_gen_kwargs[0]

        if flattened_visuals:
            image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
        else:
            image_tensor = None

        question_input = []

        for visual, context in zip(batched_visuals, contexts):
            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                image_tokens = " ".join(image_tokens)
                question = image_tokens + "\n" + context
            else:
                question = context
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            question_input.append(prompt_question)

        gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "min_new_tokens" not in gen_kwargs:
            gen_kwargs["min_new_tokens"] = 1
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        stopping_criteria = StoppingCriteriaList([Second128009Stop()])
        cont = self.model.generate(
            input_ids,
            attention_mask=attention_masks,
            pad_token_id=pad_token_ids,
            images=image_tensor,
            image_sizes=gen_kwargs["image_sizes"],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            min_new_tokens=gen_kwargs["min_new_tokens"],
            use_cache=self.use_cache,
            stopping_criteria=stopping_criteria,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        res.extend(text_outputs)
        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
        pbar.update(1)
    res = re_ords.get_original(res)

    pbar.close()
    return res


Llava.generate_until = _patched_generate_until
