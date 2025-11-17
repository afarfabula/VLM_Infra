#!/usr/bin/env python3
"""
评估管道测试脚本
用于验证各模块功能
"""

import os
import sys
import json
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

def test_data_loader():
    """测试数据加载器"""
    print("测试数据加载器...")
    
    try:
        from data_loader.vqav2_loader import create_vqav2_dataloader
        
        # 创建测试数据加载器
        data_loader = create_vqav2_dataloader(
            data_root="/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2",
            batch_size=2,
            num_workers=0
        )
        
        # 获取一个批次
        batch = next(iter(data_loader))
        
        print(f"数据加载器测试成功")
        print(f"批次类型: {type(batch)}")
        if isinstance(batch, dict):
            print(f"批次键: {list(batch.keys())}")
        
        return True
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        return False


def test_evaluator():
    """测试评估器"""
    print("测试评估器...")
    
    try:
        from evaluation.vqav2_evaluator import VQAv2Evaluator
        
        # 创建测试评估器
        evaluator = VQAv2Evaluator("/tmp/test_evaluator")
        
        # 添加测试数据
        evaluator.add_prediction(1, "cat", "cat")
        evaluator.add_prediction(2, "dog", "cat")  # 错误预测
        
        # 计算准确率
        accuracy = evaluator.calculate_accuracy()
        
        print(f"评估器测试成功")
        print(f"准确率: {accuracy:.4f}")
        
        # 保存结果
        evaluator.save_predictions()
        
        return True
        
    except Exception as e:
        print(f"评估器测试失败: {e}")
        return False


def test_distributed_utils():
    """测试分布式工具"""
    print("测试分布式工具...")
    
    try:
        from utils.distributed_utils import setup_distributed, cleanup_distributed, get_rank, get_world_size
        
        # 设置分布式环境（单进程模式）
        rank, world_size, local_rank = setup_distributed()
        
        print(f"分布式工具测试成功")
        print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        # 清理
        cleanup_distributed()
        
        return True
        
    except Exception as e:
        print(f"分布式工具测试失败: {e}")
        return False


def test_config_file():
    """测试配置文件"""
    print("测试配置文件...")
    
    try:
        config_path = "configs/vqav2_config.json"
        
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"配置文件测试成功")
        print(f"管道名称: {config['evaluation_pipeline']['name']}")
        print(f"GPU数量: {config['distributed_config']['num_gpus']}")
        
        # 检查模型配置
        if 'model_configs' in config:
            print(f"支持的模型: {list(config['model_configs'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"配置文件测试失败: {e}")
        return False


def test_main_script():
    """测试主脚本"""
    print("测试主脚本...")
    
    try:
        # 检查主脚本是否存在
        main_script = "main.py"
        
        if not os.path.exists(main_script):
            print(f"主脚本不存在: {main_script}")
            return False
        
        # 检查脚本语法
        with open(main_script, 'r') as f:
            code = f.read()
        
        # 简单的语法检查
        compile(code, main_script, 'exec')
        
        print(f"主脚本测试成功")
        return True
        
    except Exception as e:
        print(f"主脚本测试失败: {e}")
        return False


def test_launch_scripts():
    """测试启动脚本"""
    print("测试启动脚本...")
    
    scripts_to_test = ["run_single.sh", "run_distributed.sh"]
    
    for script in scripts_to_test:
        if not os.path.exists(script):
            print(f"启动脚本不存在: {script}")
            return False
        
        # 检查脚本权限
        if not os.access(script, os.X_OK):
            print(f"启动脚本不可执行: {script}")
            # 尝试设置执行权限
            try:
                os.chmod(script, 0o755)
                print(f"已设置执行权限: {script}")
            except Exception as e:
                print(f"设置执行权限失败: {e}")
                return False
    
    print(f"启动脚本测试成功")
    return True


def run_all_tests():
    """运行所有测试"""
    print("开始评估管道测试...")
    print("=" * 50)
    
    tests = [
        test_config_file,
        test_distributed_utils,
        test_evaluator,
        test_data_loader,
        test_main_script,
        test_launch_scripts
    ]
    
    results = []
    
    for test_func in tests:
        result = test_func()
        results.append((test_func.__name__, result))
        print("-" * 30)
    
    print("=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过! 评估管道准备就绪。")
        return True
    else:
        print("❌ 部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(Path(__file__).parent)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ 评估管道测试完成，可以开始分布式推理测试。")
        print("\n下一步操作:")
        print("1. 单进程测试: ./run_single.sh")
        print("2. 分布式测试: ./run_distributed.sh")
        print("3. 查看结果: ls -la results/")
    else:
        print("\n❌ 评估管道测试失败，请修复问题后重试。")
    
    sys.exit(0 if success else 1)