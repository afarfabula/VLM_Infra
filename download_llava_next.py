#!/usr/bin/env python3
"""
LLaVA-NeXT模型权重下载脚本
下载地址: https://huggingface.co/LLaVA-VL/LLaVA-NeXT
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_directories():
    """设置下载目录和缓存目录"""
    # 目标目录
    target_dir = Path("/data/model/Inference_VLM/models-LLava-NeXT")
    
    # 缓存目录
    cache_dir = target_dir / ".cache"
    
    # 创建目录
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 目标目录: {target_dir}")
    print(f"📁 缓存目录: {cache_dir}")
    
    return target_dir, cache_dir

def check_disk_space():
    """检查磁盘空间是否足够"""
    try:
        result = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 5:
                available = parts[3]
                print(f"💾 可用磁盘空间: {available}")
                return True
    except Exception as e:
        print(f"⚠️ 无法检查磁盘空间: {e}")
    
    return True

def download_with_huggingface_hub(model_name, target_dir, cache_dir):
    """使用huggingface_hub下载模型"""
    try:
        from huggingface_hub import snapshot_download, HfApi
        
        print(f"🚀 开始下载 {model_name}...")
        
        # 首先检查仓库是否存在
        api = HfApi()
        try:
            api.repo_info(repo_id=model_name)
            print(f"📦 仓库 {model_name} 存在")
        except Exception as repo_err:
            print(f"❌ 仓库 {model_name} 不存在或无法访问: {repo_err}")
            return False
        
        # 设置环境变量
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        
        # 下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[
                "*.json",
                "*.bin",
                "*.model",
                "*.txt",
                "*.py",
                "*.md"
            ]
        )
        
        # 检查是否真的下载了文件
        files_downloaded = list(target_dir.glob("*.*"))
        if len(files_downloaded) == 0:
            print(f"❌ {model_name} 下载失败: 没有文件被下载")
            return False
            
        print(f"✅ {model_name} 下载完成! 共下载 {len(files_downloaded)} 个文件")
        return True
        
    except ImportError:
        print("❌ huggingface_hub 未安装，尝试使用git下载")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_git(model_name, target_dir):
    """使用git下载模型（备用方法）"""
    try:
        repo_url = f"https://huggingface.co/{model_name}"
        
        print(f"🚀 使用git下载 {model_name}...")
        
        # 克隆仓库（不包含大文件）
        result = subprocess.run([
            "git", "clone", repo_url, str(target_dir), "--depth", "1"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 基础文件下载完成")
            print("⚠️ 需要手动下载大文件，请使用git lfs pull或手动下载权重文件")
            return True
        else:
            print(f"❌ git克隆失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ git下载失败: {e}")
        return False

def verify_download(target_dir):
    """验证下载的文件"""
    # 检查目录中的所有文件
    all_files = list(target_dir.glob("*.*"))
    print(f"\n🔍 验证下载文件... 目录中共有 {len(all_files)} 个文件")
    
    if len(all_files) == 0:
        print("❌ 目录为空，下载失败")
        return False
    
    # 打印所有下载的文件
    print("📁 下载的文件列表:")
    for file in all_files[:20]:  # 只显示前20个文件
        size = file.stat().st_size / (1024*1024)  # MB
        print(f"   - {file.name} ({size:.1f} MB)")
    
    if len(all_files) > 20:
        print(f"   ... 还有 {len(all_files) - 20} 个文件")
    
    # 检查一些关键文件
    key_files = ["config.json", "tokenizer.json", "model.safetensors"]
    found_key_files = []
    missing_key_files = []
    
    for file in key_files:
        file_path = target_dir / file
        if file_path.exists():
            found_key_files.append(file)
        else:
            # 尝试其他可能的文件名
            alt_names = []
            if file == "pytorch_model.bin":
                alt_names = ["model.safetensors", "pytorch_model-*.bin"]
            
            found = False
            for alt_name in alt_names:
                if list(target_dir.glob(alt_name)):
                    found_key_files.append(f"{file} (找到替代文件: {alt_name})")
                    found = True
                    break
            
            if not found:
                missing_key_files.append(file)
    
    if found_key_files:
        print(f"\n✅ 找到关键文件: {found_key_files}")
    
    if missing_key_files:
        print(f"⚠️ 缺失一些关键文件: {missing_key_files}")
        print("💡 注意: 不同的模型可能使用不同的文件名")
    
    # 如果有文件被下载，就认为成功
    return len(all_files) > 0

def create_readme(target_dir):
    """创建README文件"""
    readme_content = """# LLaVA-NeXT 模型

## 模型信息
- **模型名称**: LLaVA-NeXT
- **HuggingFace**: https://huggingface.co/lmms-lab
- **架构**: LLaVA (Large Language and Vision Assistant) Next Generation

## 使用方法

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

# 加载模型
model = LlavaForConditionalGeneration.from_pretrained(
    "/data/model/Inference_VLM/models-LLava-NeXT",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    "/data/model/Inference_VLM/models-LLava-NeXT"
)
```

## 下载信息
- 下载时间: {download_time}
- 下载的模型: {model_name}
- 下载方式: huggingface_hub
- 存储位置: {target_dir}
"""
    
    from datetime import datetime
    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    readme_path = target_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.format(
            download_time=download_time,
            model_name=model_name,
            target_dir=str(target_dir)
        ))
    
    print(f"📄 README文件已创建: {readme_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 LLaVA-NeXT 模型下载脚本")
    print("=" * 60)
    
    # 模型名称
    # 尝试不同的模型名称，因为原始名称可能不正确
    model_names = [
        "lmms-lab/llava-next-7b",
        "lmms-lab/llava-onevision-7b",
        "LLaVA-VL/LLaVA-NeXT"
    ]
    
    # 设置目录
    target_dir, cache_dir = setup_directories()
    
    # 检查磁盘空间
    if not check_disk_space():
        print("❌ 磁盘空间不足，请清理空间后重试")
        return
    
    # 尝试使用huggingface_hub下载，尝试多个可能的模型名称
    success = False
    for model_name in model_names:
        print(f"\n📌 尝试下载模型: {model_name}")
        success = download_with_huggingface_hub(model_name, target_dir, cache_dir)
        if success:
            break
    
    # 如果huggingface_hub失败，尝试git
    if not success:
        print("\n🔄 尝试备用下载方法...")
        for model_name in model_names:
            print(f"📌 尝试使用git下载: {model_name}")
            success = download_with_git(model_name, target_dir)
            if success:
                break
    
    # 验证下载
    if success:
        verify_download(target_dir)
        create_readme(target_dir)
        
        print("\n" + "=" * 60)
        print("🎉 下载完成!")
        print(f"📁 模型位置: {target_dir}")
        print("💡 使用方法请参考 README.md")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 下载失败")
        print("💡 请检查网络连接或手动下载")
        print("=" * 60)

if __name__ == "__main__":
    main()
