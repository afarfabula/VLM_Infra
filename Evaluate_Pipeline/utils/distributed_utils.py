#!/usr/bin/env python3
"""
分布式工具函数
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed():
    """设置分布式环境"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 如果已经初始化，直接返回
    if dist.is_initialized():
        return rank, world_size, local_rank
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    
    print(f"分布式环境初始化完成 - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("分布式环境已清理")


def is_distributed() -> bool:
    """检查是否在分布式环境中"""
    return dist.is_initialized() and int(os.environ.get('WORLD_SIZE', 1)) > 1


def get_rank() -> int:
    """获取当前进程的rank"""
    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:
    """获取进程总数"""
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """获取本地rank"""
    return int(os.environ.get('LOCAL_RANK', 0))


def barrier():
    """分布式同步屏障"""
    if is_distributed():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程上对tensor进行平均"""
    if not is_distributed():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    
    return tensor


def gather_tensors(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """在所有进程上收集tensor"""
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    rank = get_rank()
    
    # 创建收集缓冲区
    if rank == 0:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None
    
    # 收集所有tensor
    dist.gather(tensor, gathered_tensors, dst=0)
    
    if rank == 0:
        return torch.cat(gathered_tensors)
    else:
        return None


def print_rank0(*args, **kwargs):
    """只在rank 0进程打印"""
    if get_rank() == 0:
        print(*args, **kwargs)


def log_rank0(message: str):
    """只在rank 0进程记录日志"""
    if get_rank() == 0:
        print(f"[RANK 0] {message}")


def setup_logging(log_dir: str, rank: int):
    """设置分布式日志"""
    import logging
    from pathlib import Path
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"rank_{rank}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [Rank {rank}] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


if __name__ == "__main__":
    # 测试分布式工具
    rank, world_size, local_rank = setup_distributed()
    
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    print(f"Is Distributed: {is_distributed()}")
    
    # 测试同步
    barrier()
    print(f"Rank {rank}: 同步完成")
    
    # 测试rank 0打印
    print_rank0(f"这是来自rank 0的消息")
    
    cleanup_distributed()