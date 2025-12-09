"""
_glimpse_forward 方法详细注释版本

该文件包含了对 LlavaLlamaForCausalLM_GP 类中 _glimpse_forward 方法的完整注释，
用于帮助理解 GlimpsePrune 技术中的核心前向传播逻辑。
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import ModelOutput

class LlavaGPOutputWithPast(ModelOutput):
    """
    GlimpsePrune 模型的输出类，继承自 ModelOutput
    """
    logits: torch.FloatTensor = None
    le_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    input_ids: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    grid_hw: Optional[Tuple[int, int]] = None
    image_token_mask_logits: Optional[torch.Tensor] = None
    image_token_bool_masks: Optional[torch.Tensor] = None


def _glimpse_forward_annotated(
    self,
    input_ids,
    inputs_embeds,
    labels,
    attention_mask,
    position_ids,
    past_key_values,
    ref_token_masks,
    use_cache,
    image_info,
    grid_hw,
    return_dict,
    delay_selection,
    use_ref_masks,
):
    """
    GlimpsePrune 核心前向传播方法，用于处理多模态输入并执行图像标记选择。
    
    该方法的主要功能包括：
    1. 验证输入参数并进行预处理
    2. 添加可学习嵌入（LE）到输入序列
    3. 准备解码器所需的其他输入（如因果掩码）
    4. 在指定层提取注意力权重
    5. 解码图像标记掩码 logits
    6. 根据策略裁剪输入序列和缓存
    7. 执行后续的前向传播或延迟选择
    
    参数说明:
        input_ids: 输入标记 ID 张量，形状为 [batch_size, seq_length]
        inputs_embeds: 输入嵌入张量，形状为 [batch_size, seq_length, hidden_size]
        labels: 标签张量，用于计算损失，形状为 [batch_size, seq_length]
        attention_mask: 注意力掩码，指示有效位置，形状为 [batch_size, seq_length]
        position_ids: 位置 ID，指示每个标记的位置，形状为 [batch_size, seq_length]
        past_key_values: 缓存的键值对，用于加速推理
        ref_token_masks: 参考标记掩码，用于参考选择策略
        use_cache: 是否使用缓存进行推理
        image_info: 图像相关信息字典
        grid_hw: 图像网格的高度和宽度元组 (height, width)
        return_dict: 是否以字典形式返回结果
        delay_selection: 是否延迟图像标记选择
        use_ref_masks: 是否使用参考掩码
        
    返回值:
        根据 delay_selection 和 return_dict 参数返回不同的结果：
        - 如果 delay_selection 为 True，返回 LlavaGPOutputWithPast 对象或元组
        - 如果 delay_selection 为 False，调用 _glimpse_forward_after_reduction 并返回其结果
    """
    
    # 步骤 1. 检查输入参数
    # 确保分词器的填充方式是左侧填充
    assert getattr(self.config, "tokenizer_padding_side", "left") == "left"
    
    # 检查梯度检查点设置与缓存使用的兼容性
    if self.model.gradient_checkpointing and self.model.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
            
    # 获取批次大小和序列长度
    batch_size, seq_length = input_ids.shape[:2]
    past_key_values_length = 0
    # 处理过去的键值缓存
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)
        
    # 如果没有提供位置 ID，则创建默认的位置 ID
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)
    
    # 如果没有提供输入嵌入，则从输入 ID 生成嵌入
    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        
    # 步骤 2. 为可学习嵌入（LE）追加 input_ids、inputs_embeds、position_ids、attention_mask
    # 如果不使用参考掩码，则添加可学习嵌入
    if not use_ref_masks:
        input_ids, inputs_embeds, labels, position_ids, attention_mask = self._append_le(
            input_ids, inputs_embeds, labels, position_ids, attention_mask
        )
    
    # 步骤 3. 准备解码器的其他输入
    # 更新因果掩码以适应当前输入
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_key_values)
    hidden_states = inputs_embeds  # 初始化隐藏状态为输入嵌入
    
    batch_size, seq_length = input_ids.shape[:2]
    # q_indices 决定提取注意力的位置，kv_mask 决定哪些位置是图像标记
    if labels is None:
        # 如果没有标签，则将查询索引设为序列最后一个位置
        q_indices = [seq_length - 1] * batch_size
    else:
        # 如果有标签，则找到第一个非忽略标签的位置作为查询索引
        label_mask = labels != -100
        q_indices = label_mask.int().argmax(dim=-1) - 1
        q_indices = q_indices.tolist()
    # 标记图像标记的位置
    kv_mask = input_ids == IMAGE_TOKEN_INDEX
    
    # 获取配置中选定的层用于注意力提取
    selected_layers = tuple(self.config.selected_layers)
    if len(selected_layers) == 0:
        # 如果没有选定层，则不提取注意力
        batched_attns = None
    else:
        # 初始化存储注意力权重的结构
        batched_attns = [[None] * len(selected_layers) for _ in range(batch_size)]
    
    # 步骤 4. 在 LLM 前向传播过程中提取注意力
    next_decoder_cache = None
    # 确定最大前向传播层数
    max_forward_layer = max(self.config.selected_layers) if len(self.config.selected_layers) > 0 else 0
    max_forward_layer = max(max_forward_layer, self.config.reduce_layer)
    # 如果提供了标签，则总是前向传播所有层
    if labels is not None:
        max_forward_layer = len(self.model.layers) - 1  
    
    # 用于存储需要进行缩减的隐藏状态和 KV 缓存
    hidden_states_for_reduction = None
    kv_cache_for_reduction = None
    
    # 遍历模型的所有解码层
    for layer_id, decoder_layer in enumerate(self.model.layers):
        # 在第 0 层之后尝试添加可学习嵌入（第 0 层已添加过）
        if layer_id > 0 and not use_ref_masks:  
            hidden_states = self._try_add_le(layer_id, hidden_states, q_indices)
        
        # 检查当前层是否是需要提取注意力的选定层
        try:
            layer_pos = self.config.selected_layers.index(layer_id)
        except ValueError:
            layer_pos = None
        
        # 设置是否输出注意力权重
        if layer_pos is not None and not use_ref_masks:
            _output_attentions = True
        else:
            _output_attentions = False
            
        # 执行解码层前向传播（考虑梯度检查点）
        if self.model.gradient_checkpointing and self.model.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                _output_attentions,
                use_cache,
                q_indices,
                kv_mask,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=_output_attentions,
                use_cache=use_cache,
                q_indices=q_indices,
                kv_mask=kv_mask
            )
        
        # 更新隐藏状态和缓存
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[-1]
        
        # 存储提取的注意力权重
        if _output_attentions:
            attn_weights = layer_outputs[1]
            for b in range(batch_size):
                batched_attns[b][layer_pos] = attn_weights[b]
        
        # 在指定的缩减层保存隐藏状态和缓存用于后续处理
        if hidden_states_for_reduction is None:
            if layer_id == self.config.reduce_layer and layer_id < len(self.model.layers) - 1:
                if layer_id >= max_forward_layer:
                    # 如果已经到达最大前向层，则直接使用
                    hidden_states_for_reduction = hidden_states
                    kv_cache_for_reduction = next_decoder_cache if use_cache else None
                else:
                    # 否则克隆一份用于缩减处理
                    hidden_states_for_reduction = hidden_states.clone()
                    if use_cache:
                        kv_cache_for_reduction = DynamicCache()
                        for layer_idx, (key_cache, value_cache) in enumerate(next_decoder_cache):
                            kv_cache_for_reduction.update(key_cache.clone(), value_cache.clone(), layer_idx)
                    else:
                        kv_cache_for_reduction = None
        # 如果达到最大前向层则提前结束循环
        if layer_id >= max_forward_layer:
            break
    
    # 如果前向传播了所有层，则应用 LayerNorm
    if max_forward_layer >= len(self.model.layers) - 1:
        hidden_states = self.model.norm(hidden_states)
    
    # 如果未设置用于缩减的隐藏状态，则使用最终的隐藏状态和缓存
    if hidden_states_for_reduction is None:
        hidden_states_for_reduction = hidden_states
        kv_cache_for_reduction = next_decoder_cache if use_cache else None

    # 计算可学习嵌入的损失（如果有标签）
    if labels is not None:
        le_logits = self.lm_head(hidden_states).float()
        shift_le_logits = le_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        shift_le_logits = shift_le_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_le_logits.device)
        le_loss = loss_fct(shift_le_logits, shift_labels)
        del le_logits, shift_le_logits, shift_labels
    else:
        le_loss = None
    
    # 清理临时变量释放内存
    del hidden_states
    del next_decoder_cache
    
    # 步骤 5. 解码图像标记
    # 整理提取的注意力权重
    if batched_attns is not None and not use_ref_masks:
        for i, one_attns in enumerate(batched_attns):
            batched_attns[i] = torch.stack(one_attns, dim=1)  # [num_tokens, num_layers, num_heads]
        batched_attns = torch.stack(batched_attns, dim=0)  # [bsz, num_tokens, num_layers, num_heads]
        batched_attns = batched_attns.flatten(2)  # [bsz, num_tokens, num_layers * num_heads]
        
    # 根据不同策略获取图像标记掩码 logits
    if use_ref_masks:
        # 使用参考掩码
        image_token_mask_logits = []
        for b in range(batch_size):
            image_token_mask_logits.append(torch.logit(ref_token_masks[b].float().to(device=hidden_states_for_reduction.device).view(1, -1)))
    elif self.config.use_zero_masks:
        # 使用零掩码
        image_token_mask_logits = []
        for b in range(batch_size):
            image_token_mask_logits.append(torch.logit(torch.zeros((1, grid_hw[0] * grid_hw[1]), device=hidden_states_for_reduction.device)))
    else:
        # 使用注意力融合器解码图像标记掩码 logits
        image_token_mask_logits = self._decode_image_token_mask_logits(batched_attns, grid_hw, **image_info)

    # 步骤 6. 裁剪（缩减的图像标记和 LE 标记）hidden_states、next_cache、input_ids、attention_mask
    if not use_ref_masks:
        # 移除可学习嵌入部分
        le_length = self.config.le_length
        input_ids = input_ids[:, :-le_length]
        inputs_embeds = inputs_embeds[:, :-le_length, :]
        hidden_states_for_reduction = hidden_states_for_reduction[:, :-le_length]
        # 裁剪 KV 缓存
        if kv_cache_for_reduction is not None:
            try:
                kv_cache_for_reduction.crop(-le_length)
            except AttributeError:
                self._crop_kv_cache(kv_cache_for_reduction, -le_length)
            
        position_ids = position_ids[:, :-le_length]
        attention_mask = attention_mask[:, :-le_length]

    # 根据 delay_selection 决定是返回中间结果还是继续处理
    if delay_selection:
        # 延迟选择：保存中间结果并返回
        self.todo_selection = True
        logits = None
        rtn_dict = LlavaGPOutputWithPast(
                logits=logits,
                le_loss=le_loss,
                past_key_values=kv_cache_for_reduction,
                hidden_states=hidden_states_for_reduction,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                grid_hw=grid_hw,
                image_token_mask_logits=image_token_mask_logits,
        )
        self.glimpse_return_before_selection = rtn_dict

        if return_dict:
            return rtn_dict
        else:
            return (
                logits,
                le_loss,
                kv_cache_for_reduction,
                hidden_states_for_reduction,
                input_ids,
                inputs_embeds,
                attention_mask,
                position_ids,
                grid_hw,
                image_token_mask_logits,
            )
    else:
        # 立即选择：缩减标记并继续前向传播
        reduced_info = self._reduce_tokens(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states_for_reduction,
            past_key_values=kv_cache_for_reduction,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_token_mask_logits=image_token_mask_logits,
            grid_hw=grid_hw,
        )
        return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)