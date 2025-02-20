import os
import torch
from .model import Model
from .arguments import Config
from .environment import state, device
from minimax.algorithm import get_all_moves
from halma.constants import BLACK


# 使用基于 PPO 算法的模型，选择最佳的动作
def ppo_move(board, game):
    hyp = Config()
    agent = Model(hyp).to()

    # 如果存在模型文件
    if os.path.exists(hyp.model_path):
        # 加载模型权重
        agent.load_state_dict(torch.load(f'{hyp.model_path}/state_dict.pt', map_location=device))

    # 游戏状态【state, action】
    X = state([board])
    # 所有可能的动作序列
    boards = get_all_moves(board, BLACK, game)
    # agent 选择动作【**X = state, action]
    action = agent(**X)[:, :len(boards)].argmax(1).item()

    return boards[action]
