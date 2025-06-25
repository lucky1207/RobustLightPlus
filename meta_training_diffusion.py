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
import copy  # 新增，用于深拷贝模型参数
from typing import List

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

def create_replay_buffer(memory_data, device):
    """根据memory原始数据创建ReplayBuffer（保持与旧流程一致）"""
    states, actions, next_states, pressure_rewards, q_length_rewards = memory_data
    list_feature = [
        "traffic_movement_pressure_queue_efficient",
        "lane_run_in_part",
    ]
    states = process_feature(states, list_feature)
    next_states = process_feature(next_states, list_feature)

    actions = np.eye(4)[np.array(actions)].reshape(240, states.shape[1], 4)
    q_length_rewards = np.array(q_length_rewards).reshape(240, states.shape[1], 1)

    replay_buffer = ReplayBuffer((states, actions, next_states, q_length_rewards), 240, device)
    return replay_buffer

def meta_train_diffusion(
    config: dict,
    outer_memory_paths: List[str],
    inner_memory_path: str,
    output_dir: str,
    device: torch.device,
):
    """基于 Reptile 思想的简化版 MAML 训练。"""
    logger.info("开始 Meta-Learning 训练 (Reptile 简化版)")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ==== 1. 构建 ReplayBuffer ====
    outer_buffers: List[ReplayBuffer] = []
    for p in outer_memory_paths:
        if not os.path.exists(p):
            logger.error(f"外循环 memory 文件不存在: {p}")
            raise FileNotFoundError(p)
        mb_data = load_memory_data(p)
        outer_buffers.append(create_replay_buffer(mb_data, device))
        logger.info(f"  已加载 outer task: {p}")

    if not os.path.exists(inner_memory_path):
        logger.error(f"内循环 memory 文件不存在: {inner_memory_path}")
        raise FileNotFoundError(inner_memory_path)

    inner_buffer = create_replay_buffer(load_memory_data(inner_memory_path), device)
    logger.info(f"  已加载 inner task (目标): {inner_memory_path}")

    # ==== 2. 初始化 Predictor ====
    diffusion_predictor = Diffusion_Predictor(
        state_dim=inner_buffer.states.shape[-1],
        action_dim=inner_buffer.actions.shape[-1],
        device=device,
        config=config,
        log_writer=False,
    )
    logger.info("Diffusion Predictor 初始化完成")

    # ==== 3. Meta 参数 ====
    meta_iterations = 1000  # 外循环次数
    inner_steps = 1  # 每个外任务的适应步数
    meta_lr = 0.1

    use_wandb = setup_wandb(config, project_name="RobustLightPlus-Diffusion-Meta")

    for meta_step in tqdm(range(meta_iterations), desc="Meta 训练进度"):
        meta_state = copy.deepcopy(diffusion_predictor.predictor.state_dict())
        delta_state = {k: torch.zeros_like(v) for k, v in meta_state.items()}

        # ---- Inner loop over outer tasks ----
        for buf in outer_buffers:
            diffusion_predictor.predictor.load_state_dict(meta_state)
            diffusion_predictor.ema_model.load_state_dict(meta_state)

            diffusion_predictor.train(
                replay_buffer=buf,
                iterations=inner_steps,
                batch_size=config["batch_size"],
                log_writer=False,
            )

            adapted_state = diffusion_predictor.predictor.state_dict()
            for k in delta_state:
                delta_state[k] += adapted_state[k] - meta_state[k]

        # ---- Meta update (Reptile) ----
        for k in meta_state:
            meta_state[k] += meta_lr * delta_state[k] / len(outer_buffers)

        diffusion_predictor.predictor.load_state_dict(meta_state)
        diffusion_predictor.ema_model.load_state_dict(meta_state)
        diffusion_predictor.predictor_optimizer = torch.optim.Adam(
            diffusion_predictor.predictor.parameters(), lr=config["lr"]
        )

        # 在 inner task 上做一次训练 / 评估
        metric = diffusion_predictor.train(
            replay_buffer=inner_buffer,
            iterations=1,
            batch_size=config["batch_size"],
            log_writer=use_wandb,
        )

        if use_wandb and (meta_step % 50 == 0):
            wandb.log({
                "Meta/Step": meta_step,
                "Meta/Inner_Loss": metric["pred_loss"][-1] if metric["pred_loss"] else 0,
            })

        if meta_step % 100 == 0:
            logger.info(f"[Meta-Step {meta_step}] Inner Loss: {metric['pred_loss'][-1] if metric['pred_loss'] else 0}")

    # ==== 4. 保存 Meta 训练模型 ====
    final_path = os.path.join(output_dir, "diffusion_model_meta_final.pth")
    diffusion_predictor.save_model(final_path)
    logger.info(f"Meta 训练结束，模型保存在 {final_path}")

    if use_wandb:
        wandb.finish()


# ---------------- 单任务普通训练 ----------------

def train_diffusion(
    config: dict,
    memory_path: str,
    output_dir: str,
    device: torch.device,
):
    """保持与原来一致的单任务 Diffusion 训练流程"""
    logger.info("开始单任务 Diffusion 训练")

    os.makedirs(output_dir, exist_ok=True)

    replay_buffer = create_replay_buffer(load_memory_data(memory_path), device)

    diffusion_predictor = Diffusion_Predictor(
        state_dim=replay_buffer.states.shape[-1],
        action_dim=replay_buffer.actions.shape[-1],
        device=device,
        config=config,
        log_writer=False,
    )

    use_wandb = setup_wandb(config)

    max_timesteps = 5000
    batch_size = config["batch_size"]

    for step in tqdm(range(max_timesteps), desc="训练进度"):
        metric = diffusion_predictor.train(
            replay_buffer=replay_buffer,
            iterations=1,
            batch_size=batch_size,
            log_writer=use_wandb,
        )

        if use_wandb and (step % 100 == 0):
            wandb.log({
                "Training/Step": step,
                "Training/Loss": metric["pred_loss"][-1] if metric["pred_loss"] else 0,
                "Training/LR": diffusion_predictor.predictor_optimizer.param_groups[0]["lr"],
            })

    final_path = os.path.join(output_dir, "diffusion_model_final.pth")
    diffusion_predictor.save_model(final_path)
    logger.info(f"单任务训练完成，模型保存在 {final_path}")

    if use_wandb:
        wandb.finish()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练Diffusion模型")
    parser.add_argument("--meta", action="store_true", default=True, help="使用 Meta-Learning 训练模式")
    parser.add_argument("--outer_memory_paths", type=str,
                        default="memory/data_jn_2.pkl,memory/data_jn_3.pkl,memory/data_hz_1.pkl,memory/data_hz_2.pkl",
                        help="逗号分隔的外循环 memory 文件路径列表")
    parser.add_argument("--inner_memory_path", type=str, default="memory/data_jn_1.pkl",
                        help="内循环 (目标) memory 文件路径")
    parser.add_argument("--memory_path", type=str, default="memory/data_jn_1.pkl",
                        help="普通单任务训练 memory 文件路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints/diffusion",
                        help="输出目录")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 wandb 日志")
    
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
    if (not args.meta) and (not os.path.exists(args.memory_path)):
        logger.error(f"Memory文件不存在: {args.memory_path}")
        return
    
    # 打印配置
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 开始训练
    try:
        if args.meta:
            outer_paths = [p.strip() for p in args.outer_memory_paths.split(',') if p.strip()]
            meta_train_diffusion(config, outer_paths, args.inner_memory_path, args.output_dir, device)
        else:
            train_diffusion(config, args.memory_path, args.output_dir, device)
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
