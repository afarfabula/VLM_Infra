---
license: apache-2.0
task_categories:
- image-text-to-text
language:
- en
tags:
- visual question answering
---

# VisCoT Dataset Card

![dataset](assets/dataset.png)

![dataset](assets/dataset_gqa.png)

There is a shortage of multimodal datasets for training multi-modal large language models (MLLMs) that require to identify specific regions in an image for additional attention to improve response performance. This type of dataset with grounding bbox annotations could possibly help the MLLM output intermediate interpretable attention area and enhance performance.
To fill the gap, we curate a visual CoT dataset. **This dataset specifically focuses on identifying critical regions within images, a feature essential for models to concentrate on relevant visual elements to improve response accuracy. Each data sample consists of a question, answer, and a corresponding visual bounding box across five domains. Some data samples also include extra detailed reasoning steps.**

To ensure a robust foundation for detailed visual and textual analysis, our dataset deliberately integrates a diverse selection of data including **text/doc, fine-grained understanding, charts, general VQA, and relation reasoning**. These data domains are deliberately chosen to cultivate a comprehensive skill set across varied analytical tasks: 1) Text/doc enhances MLLM's capabilities on OCR and contextual understanding, crucial for applications requiring text interpretation in complex environments. 2) Fine-grained understanding aids in identifying and distinguishing subtle differences in visual appearance and patterns. 3) Charts foster the ability to interpret graphical data, which are essential for business and scientific applications. 4) General VQA exposes models to a wide array of visual queries, improving their general usability. 5) Relation reasoning data develops spatial and contextual awareness of MLLMs, vital for interactive and navigational tasks. Together, these modalities ensure the dataset not only fills existing gaps but also enhances the versatility and contextual awareness of MLLMs across varied scenarios.

## Dataset details

- `viscot_363k.json`: the data list which only contains VisCoT-related training data
- `viscot_mixed_2m.json`: the mixed data list for reproducing the VisCoT
- `metadata/`: metainfo folder, including more raw and detailed information and annotations
  - `cub_cot_train.jsonl`: metainfo for the CUB dataset
  - `docvqa_cot_train.jsonl`: metainfo for the DocVQA dataset
  - ...

**Dataset date:**
VisCoT-1.0 Dataset was collected in June 2024. 

**Paper or resources for more information:**

Github: https://github.com/deepcs233/Visual-CoT

Paper: https://arxiv.org/abs/2403.16999

**License:**
Attribution-NonCommercial 4.0 International

**Where to send questions or comments about the model:**
https://github.com/deepcs233/Visual-CoT/issues

## Disclaimer

This dataset was collected and released solely for research purposes, with the goal of making the MLLMs dynamically focus on visual inputs and provide intermediate interpretable thoughts. The authors are strongly against any potential harmful use of the data or technology to any party.

### Intended Use

The data, code, and model checkpoints are intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper. The data, code, and model checkpoints are not intended to be used in clinical care or for any clinical decision making purposes.

### Primary Intended Use
The primary intended use is to support AI researchers reproducing and building on top of this work. \shortname{} and its associated models should be helpful for exploring various vision question answering (VQA) research questions.

### Out-of-Scope Use
Any deployed use case of the model --- commercial or otherwise --- is out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are intended for research use only and not intended for deployed use cases.