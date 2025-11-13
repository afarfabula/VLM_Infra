# LLaVA 启动 VLM 对话指令

本文整理了两种常用的启动方式：
- CLI 单图对话（简单、直接，单张图片）
- Gradio Web UI（分布式：控制器 + 模型工作进程 + Web 界面）

建议在不同终端窗口分别运行需要常驻的进程（控制器、worker、web），并保证 `CUDA` 环境可用。


## 一、CLI：单图对话（命令行）

最简单的方式，适合快速验证模型与图片的结合。

- 命令：
‘‘‘
CUDA_VISIBLE_DEVICES=7 python -m llava.serve.cli --model-path liuhaotian/llava-v1.5-7b --image-file "https://llava-vl.github.io/static/images/view.jpg" --load-4bit

’’’


```
python -m llava.serve.cli \
  --model-path <MODEL_PATH> \
  --image-file <IMAGE_PATH> \
  --device cuda \
  --temperature 0.2 \
  --max-new-tokens 512
```

- 参数说明：
  - `--model-path`：模型权重路径或 HuggingFace 名称（例如：`liuhaotian/llava-v1.5-7b`）。
  - `--image-file`：待对话的图片路径（支持本地路径或 HTTP/HTTPS 链接）。
  - `--device`：推理设备（如 `cuda`、`cuda:0`、`cpu`）。
  - `--temperature`：采样温度，越高越随机。
  - `--max-new-tokens`：单次生成的最大 token 数。
  - 可选：`--load-8bit` 或 `--load-4bit`（低精度加载，节省显存）。

- 交互特性：
  - 程序启动后会提示输入；第一轮对话自动插入图片 token。
  - 按回车输入空行即可退出。

示例：
```
python -m llava.serve.cli \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-file llava/serve/examples/waterview.jpg \
  --device cuda:0 \
  --temperature 0.2 \
  --max-new-tokens 512
```


## 二、Gradio Web UI：控制器 + 模型Worker + Web界面

此模式适合多人并发和更友好的可视化交互。

### 1）启动控制器（Controller）
在终端 A 中运行：
```
python -m llava.serve.controller --host 0.0.0.0 --port 21001 --dispatch-method shortest_queue
```

### 2）启动模型 Worker（Model Worker）
在终端 B 中运行：
```
python -m llava.serve.model_worker \
  --host 0.0.0.0 \
  --port 21002 \
  --worker-address http://localhost:21002 \
  --controller-address http://localhost:21001 \
  --model-path <MODEL_PATH> \
  --model-base <MODEL_BASE> \
  --device cuda:0 \
  --limit-model-concurrency 5 \
  --load-8bit
```
- 说明：
  - `--model-path`：同 CLI。
  - `--model-base`：如果模型是增量权重（如 LoRA），可指定底座模型；否则可省略。
  - `--device`：可指定 `cuda:0`、`cuda:1` 等，挂到指定 GPU。
  - `--limit-model-concurrency`：并发请求数量上限（队列控制）。
  - `--load-8bit/--load-4bit`：低精度加载选项，按需选择。

### 3）启动 Web 界面（Gradio）
在终端 C 中运行：
```
python -m llava.serve.gradio_web_server \
  --host 0.0.0.0 \
  --port 7860 \
  --controller-url http://localhost:21001 \
  --model-list-mode once \
  --share
```
- 浏览器访问：`http://localhost:7860/`
- 页面左侧选择已注册的模型（来自 worker 注册），上传图片并输入问题即可开始多轮对话。


## 三、常用模型与配置

- `--model-path` 常见示例：
  - `liuhaotian/llava-v1.5-7b`
  - 本地路径：`/path/to/your/llava/model`

- 视觉塔（Vision Tower）与图片预处理：由内部自动加载（`transformers.CLIPVisionModel` + `CLIPImageProcessor`），无需手动指定；图片尺寸与 patch 数等在模型内部处理。


## 四、注意事项与常见问题

- 显存与上下文长度：
  - 多轮对话会累积 KV Cache，占用显存并受 `max_position_embeddings`（常见 2048）限制。
  - 若出现超长序列警告（如 `2406 > 2048`），请重启会话或缩短历史。

- 单图对话限制：
  - CLI 方式（`llava.serve.cli`）为单图；第一轮对话自动插入图片 token。

- 多图能力：
  - 当前经典 LLaVA-1.5 CLI 不支持同时持有多张图做跨图对比；高级多图功能见更高版本或特定前端。

- GPU 选择：
  - 在 worker 启动命令中通过 `--device cuda:N` 指定具体 GPU。
  - 或设置环境变量 `CUDA_VISIBLE_DEVICES` 控制可见设备。

- 低精度加载：
  - `--load-8bit` / `--load-4bit` 可显著降低显存占用，但略有精度影响。


## 五、快速复用命令摘录

- CLI：
```
python -m llava.serve.cli --model-path <MODEL_PATH> --image-file <IMAGE_PATH> --device cuda --temperature 0.2 --max-new-tokens 512
```

- Controller：
```
python -m llava.serve.controller --host 0.0.0.0 --port 21001
```

- Worker：
```
python -m llava.serve.model_worker --controller-address http://localhost:21001 --worker-address http://localhost:21002 --model-path <MODEL_PATH> --device cuda:0
```

- Web：
```
python -m llava.serve.gradio_web_server --host 0.0.0.0 --port 7860 --controller-url http://localhost:21001
```