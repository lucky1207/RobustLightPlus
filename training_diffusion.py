#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于state_filling的Diffusion训练算法
加载memory数据进行训练
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
from loguru import logger
import wandb
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import DMBP_config
from utils.batch_buffer import ReplayBuffer
from inferences.state_filling import Diffusion_Predictor


def load_memory_data(memory_path):
    """
    加载memory数据
    
    Args:
        memory_path: memory数据文件路径
        
    Returns:
        states, actions, next_states, rewards: 加载的数据
    """
    logger.info(f"正在加载memory数据: {memory_path}")
    
    try:
        with open(memory_path, 'rb') as f:
            memory_data = pickle.load(f)
        
        # 假设数据格式为 (states, actions, next_states, rewards)
        if len(memory_data) == 5:
            states, actions, next_states, pressure_rewards, q_length_rewards = memory_data
    
            return states, actions, next_states, pressure_rewards, q_length_rewards
        else:
            raise ValueError("Memory数据格式不正确，期望为(states, actions, next_states, rewards)")
            
    except Exception as e:
        logger.error(f"加载memory数据失败: {e}")
        raise


def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    return device


def setup_wandb(config, project_name="RobustLightPlus-Diffusion"):
    """设置wandb日志"""
    try:
        wandb.init(
            project=project_name,
            config=config,
            name=f"diffusion_training_{config['type']}_{config['beta_schedule']}",
            tags=["diffusion", "state_filling", "training"]
        )
        logger.info("Wandb初始化成功")
        return True
    except Exception as e:
        logger.warning(f"Wandb初始化失败: {e}")
        return False

def process_feature(states, list_feature):
    fe_len = len(states[list_feature[0]])
    s = np.zeros(shape=(240, int(fe_len/240), 24))
    for i in range(len(list_feature)):
        tmp_arr = np.array(states[list_feature[i]]).reshape(240, int(fe_len/240), 12)
        s[:,:,i*12:(i+1)*12] = tmp_arr
    return s

def train_diffusion(config, memory_path, output_dir, device):
    """
    训练diffusion模型
    
    Args:
        config: 训练配置
        memory_path: memory数据路径
        output_dir: 输出目录
        device: 训练设备
    """
    logger.info("开始训练Diffusion模型")
    
    # 加载数据
    states, actions, next_states, pressure_rewards, q_length_rewards = load_memory_data(memory_path)
    list_feature = ["traffic_movement_pressure_queue_efficient",
                "lane_run_in_part"]
    states = process_feature(states, list_feature)
    next_states = process_feature(next_states, list_feature)
  
   
    actions = np.eye(4)[np.array(actions)].reshape(240, states.shape[1], 4)
    pressure_rewards = np.array(pressure_rewards).reshape(240, states.shape[1], 1)
    q_length_rewards = np.array(q_length_rewards).reshape(240, states.shape[1], 1)
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    replay_buffer = ReplayBuffer((states, actions, next_states, q_length_rewards),
                                    240, device)

    # 初始化Diffusion Predictor
    diffusion_predictor = Diffusion_Predictor(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        config=config,
        log_writer=False
    )
    logger.info("Diffusion Predictor初始化完成")
    
    # 设置wandb
    use_wandb = setup_wandb(config)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练参数
    max_timesteps = 5000
    batch_size = config['batch_size']
    checkpoint_every = 1000
    checkpoint_start = 0
    
    logger.info(f"开始训练，最大步数: {max_timesteps}, 批次大小: {batch_size}")
    
    # 训练循环
    for step in tqdm(range(max_timesteps), desc="训练进度"):
        # 训练一个批次
        metric = diffusion_predictor.train(
            replay_buffer=replay_buffer,
            iterations=1,
            batch_size=batch_size,
            log_writer=use_wandb
        )
        
        # 记录训练指标
        if use_wandb and step % 100 == 0:
            wandb.log({
                "Training/Step": step,
                "Training/Loss": metric['pred_loss'][-1] if metric['pred_loss'] else 0,
                "Training/Learning_Rate": diffusion_predictor.predictor_optimizer.param_groups[0]['lr']
            })
        
        # 保存检查点
        # if step >= checkpoint_start and step % checkpoint_every == 0:
        #     checkpoint_path = os.path.join(output_dir, f"diffusion_checkpoint_step_{step}.pth")
        #     diffusion_predictor.save_checkpoint(checkpoint_path)
        #     logger.info(f"检查点已保存: {checkpoint_path}")
            
            # # 同时保存完整模型
            # model_path = os.path.join(output_dir, f"diffusion_model_step_{step}.pth")
            # diffusion_predictor.save_model(model_path)
            # logger.info(f"完整模型已保存: {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "diffusion_model_final.pth")
    diffusion_predictor.save_model(final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")
    
    if use_wandb:
        wandb.finish()
    
    logger.info("训练完成！")



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练Diffusion模型")
    parser.add_argument("--memory_path", type=str, default="memory/data_jn_1.pkl",
                       help="Memory数据文件路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints/diffusion",
                       help="输出目录")
      
    parser.add_argument("--no_wandb", action="store_true",
                       help="禁用wandb日志")
    
    args = parser.parse_args()
    
    # 设置日志
    logger.add("logs/training_diffusion.log", rotation="10 MB", level="INFO")
    logger.info("=" * 50)
    logger.info("Diffusion训练开始")
    logger.info("=" * 50)
    
  
    config = DMBP_config
    
    # 禁用wandb
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    
    # 设置设备
    device = setup_device()
    
    # 检查文件存在性
    if not os.path.exists(args.memory_path):
        logger.error(f"Memory文件不存在: {args.memory_path}")
        return
    
    # 打印配置
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 开始训练
    try:
        train_diffusion(config, args.memory_path, args.output_dir, device)
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
