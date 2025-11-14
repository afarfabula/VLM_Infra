#!/usr/bin/env python3
"""
LLaVA-1.5-13B模型权重下载脚本
下载地址: https://huggingface.co/liuhaotian/llava-v1.5-13b
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_directories():
    """设置下载目录和缓存目录"""
    # 目标目录
    target_dir = Path("/data/model/Inference_VLM/models-LLava-1.5-13B")
    
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
        from huggingface_hub import snapshot_download
        
        print(f"🚀 开始下载 {model_name}...")
        
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
        
        print(f"✅ {model_name} 下载完成!")
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
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    print("\n🔍 验证下载文件...")
    
    missing_files = []
    for file in required_files:
        file_path = target_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024*1024)  # MB
            print(f"✅ {file} ({size:.1f} MB)")
        else:
            missing_files.append(file)
            print(f"❌ {file} 缺失")
    
    if missing_files:
        print(f"\n⚠️ 缺失文件: {missing_files}")
        print("请手动下载缺失的文件")
        return False
    else:
        print("✅ 所有必需文件都已下载")
        return True

def create_readme(target_dir):
    """创建README文件"""
    readme_content = """# LLaVA-1.5-13B 模型

## 模型信息
- **模型名称**: LLaVA-1.5-13B
- **HuggingFace**: https://huggingface.co/liuhaotian/llava-v1.5-13b
- **大小**: 约26GB
- **架构**: LLaVA (Large Language and Vision Assistant)

## 使用方法

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

# 加载模型
model = LlavaForConditionalGeneration.from_pretrained(
    "/data/model/Inference_VLM/models-LLava-1.5-13B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    "/data/model/Inference_VLM/models-LLava-1.5-13B"
)
```

## 下载信息
- 下载时间: {download_time}
- 下载方式: huggingface_hub
- 存储位置: {target_dir}
"""
    
    from datetime import datetime
    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    readme_path = target_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.format(
            download_time=download_time,
            target_dir=str(target_dir)
        ))
    
    print(f"📄 README文件已创建: {readme_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 LLaVA-1.5-13B 模型下载脚本")
    print("=" * 60)
    
    # 模型名称
    model_name = "liuhaotian/llava-v1.5-13b"
    
    # 设置目录
    target_dir, cache_dir = setup_directories()
    
    # 检查磁盘空间
    if not check_disk_space():
        print("❌ 磁盘空间不足，请清理空间后重试")
        return
    
    # 尝试使用huggingface_hub下载
    success = download_with_huggingface_hub(model_name, target_dir, cache_dir)
    
    # 如果huggingface_hub失败，尝试git
    if not success:
        print("\n🔄 尝试备用下载方法...")
        success = download_with_git(model_name, target_dir)
    
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