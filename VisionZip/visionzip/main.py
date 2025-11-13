from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
from .clip_encoder import CLIPVisionTower_VisionZip
from .llava_arch import (
    prepare_inputs_labels_for_multimodal_visionzip,
    encode_images_visionzip,
    encode_images_visionzip_multi,
    restore_image_features_sorted,
)

def visionzip(model, dominant=191, contextual=30):
    """
    将 VisionZip 的「训练自由（training-free）」视觉 token 压缩补丁注入到现有 VLM（如 LLaVA）中，
    以在不改动/不微调任何权重的情况下，加速推理并降低视觉 token 负载。

    核心思想：
    - 对 CLIP 视觉编码器的注意力与层进行轻量级猴子补丁（forward 替换），按层次统计/筛选视觉 token；
    - 两阶段压缩：先选出 Dominant（显著）token，再进行 Contextual（上下文补齐）聚合；
    - 在 LLaVA 的多模态拼接逻辑处接入 VisionZip 的编码/还原流程，使生成仍走标准 `model.generate`。

    参数含义：
    - dominant：Dominant tokens 目标数量（包含 CLS）；实现内部通常会预留一个 CLS，因此这里传入后会使用 `dominant-1` 作为非 CLS 上限；
    - contextual：Contextual tokens 目标数量，用于补齐显著 token 的上下文信息，提升图像语义覆盖。

    返回：
    - 注入完成的 `model`（原模型实例，方法已动态替换）。
    """

    # 向视觉塔（CLIP）注入运行时信息：Dominant/Contextual 数量等。
    # 注意：这里使用 `dominant-1`，是因为 CLS token 会被保留为显著 token 之一，
    # 非 CLS 的 Dominant 选择目标需减一，以保证总数约为 dominant。
    apply_info(
        model.model.vision_tower.vision_tower,
        dominant_num=dominant - 1,
        contextual_num=contextual,
    )

    # 对 CLIP 的层与注意力进行猴子补丁：替换 forward，实现统计/筛选与压缩所需的度量与数据流。
    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    # training-free 的关键：仅替换 forward，不改动权重，不需要训练或微调。
    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    # 替换 LLaVA 中视觉塔的 forward，使其在编码阶段执行 VisionZip 的两阶段压缩策略。
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

    # 接入 LLaVA 的多模态准备/编码/还原逻辑：
    # - prepare_inputs_labels_for_multimodal_visionzip：在将图像特征与文本拼接前，执行 anyres 的恢复与排序；
    # - encode_images_visionzip(_multi)：单/多图编码入口，内部调用视觉塔（已替换）进行两阶段压缩；
    # - restore_image_features_sorted：将压缩后的特征按照原始网格顺序恢复，以适配 projector 与后续拼接。
    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
            prepare_inputs_labels_for_multimodal_visionzip
        )
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip

    # 返回原模型实例（方法已动态替换）：调用方继续使用 `model.generate` 即可触发 VisionZip。
    return model
