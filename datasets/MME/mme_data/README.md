---
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: category
    dtype: string
  splits:
  - name: test
    num_bytes: 1733070098.024
    num_examples: 2374
  download_size: 864018279
  dataset_size: 1733070098.024
---

# Evaluation Dataset for MME