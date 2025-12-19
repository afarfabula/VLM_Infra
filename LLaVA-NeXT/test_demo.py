from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from transformers import StoppingCriteria, StoppingCriteriaList

class EosStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id, *extra_eos):
        # 把 eos_token_id 和额外给的特殊 token id 全部收进一个集合
        self.eos_tokens = {eos_token_id, *extra_eos}
        self.counter = 0
        print(f"EosStoppingCriteria初始化: 停止token ids={self.eos_tokens}")

    def __call__(self, input_ids, scores, **kwargs):
        # 获取最后一个生成的token id
        last_token_id = input_ids[0, -1].item()
        # 检查是否需要停止
        stop = last_token_id in self.eos_tokens
        #print(f"EosStoppingCriteria.__call__: 最后一个token id={last_token_id}, 是否停止={stop}")
        if stop:
            self.counter += 1
        
        return stop & (self.counter >= 2)



from PIL import Image
import requests
import copy
import torch

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="flash_attention_2")

model.eval()
model.tie_weights()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image? "
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


# 确保pad_token和pad_token_id正确设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"已设置pad_token_id为: {tokenizer.pad_token_id}")

tokenizer.padding_side = "left"
pad_id = tokenizer.pad_token_id
attention_mask = (input_ids != pad_id).long()
eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

# 保存原始输入长度
original_input_length = input_ids.shape[1]
print(f"原始输入长度: {original_input_length}")

# 创建只包含128009的停止条件列表
stop_token_ids = [128009]  # 只使用128009作为停止token

# 过滤掉None值
stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

print(f"使用的停止token ids: {stop_token_ids}")

print(f"使用的eos_id: {eos_id}")
print(f"使用的pad_id: {pad_id}")

# 创建停止条件
stopping_criteria = StoppingCriteriaList([
    EosStoppingCriteria(*stop_token_ids),  # 传递所有停止token ids
   
])

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    attention_mask=attention_mask,
    pad_token_id=pad_id,
    eos_token_id=eos_id,
    do_sample=True,
    temperature=0.7,  # 调整温度参数避免重复
    top_p=0.95,  # 调整top_p参数
    min_new_tokens=5,  # 减少最小生成token数
    max_new_tokens=1024,  # 设置最大生成token数与MaxTokensStoppingCriteria保持一致
    early_stopping=False,  # 关闭early_stopping，因为num_beams=1
    stopping_criteria=stopping_criteria,
    # Modalities should be the same size as the batch size
    tokenizer=tokenizer,  
    modalities=["image"]*input_ids.shape[0]
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
