import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from .model import Model
from .environment import HalmaGameEnvironment, device
from .arguments import Config
from halma.constants import BLACK, WHITE
from minimax.algorithm import get_all_moves


# 从多元正态分布中采样动作和计算对数概率
def sample_from_dist(mean, action=None):  # Actor: 无 action；Critic: 有 action
    """
    mean:
        shape [bs, nd]
        shape [批量大小, 动作数量]

    Return:
        action: shape [bs, nd]

    """
    # actor 模型输出动作矩阵
    batch_size, action_dims = mean.shape

    # 构建多元正态分布【利用概率分布再采样动作，增加探索性】
    cov_mat = torch.diag(torch.full(size=(action_dims,), fill_value=0.5)).to(device)  # 协方差矩阵 cov_mat
    dist = MultivariateNormal(mean, cov_mat)  # mean 作为均值向量，cov_mat 作为协方差矩阵

    # 再采样，动作矩阵的值略微变化
    action = dist.sample() if action is None else action
    # 计算对数概率，用于计算策略的梯度
    log_prob = dist.log_prob(action)

    return action, log_prob


# minimax 算法，返回最佳的移动【下标】
def minimax(board, depth, max_player=False, alpha=float('-inf'), beta=float('inf')):
    # 检查是否达到了最大搜索深度，或者是否找到了赢家
    if depth == 0 or board.winner() is not None:
        # 返回棋盘评估值、棋盘状态
        return board.evaluate(), board, 0

    # 最小化玩家
    if not max_player:
        # 最小玩家试图最小化评估分数
        min_evaluation = float("inf")
        best_move = None
        best_idx = -1  # 对应移动

        # 遍历所有有效移动，评估每个移动
        for idx, move in enumerate(get_all_moves(board, BLACK)):
            # 递归下一回合【最大化玩家】
            evaluation = minimax(move, depth - 1, True)[0]

            # 更新
            if evaluation < min_evaluation:
                min_evaluation = evaluation
                best_move = move
                best_idx = idx  # 对应移动【下标】
            # 更新beta值
            beta = min(beta, evaluation)

            # 如果beta小于等于alpha，跳出循环
            if beta <= alpha:
                break
        return min_evaluation, best_move, best_idx

    # 最大化玩家
    else:
        # 最大玩家试图最大化评估分数
        max_evaluation = float("-inf")
        best_move = None
        best_idx = -1  # 对应移动

        # 遍历所有有效移动，评估每个移动
        for idx, move in enumerate(get_all_moves(board, WHITE)):
            # 递归下一回合
            evaluation = minimax(move, depth - 1, False)[0]

            # 更新
            if evaluation > max_evaluation:
                max_evaluation = evaluation
                best_move = move
                best_idx = idx  # 对应移动【下标】
            # 更新alpha值
            alpha = max(alpha, evaluation)

            # 如果beta小于等于alpha，跳出循环
            if beta <= alpha:
                break
        return max_evaluation, best_move, best_idx


# PPO 训练类
class PPOTrainer:
    def __init__(self, config: Config()):
        self.environment = HalmaGameEnvironment(config)  # PPO 环境

        self.actor = Model(config, mode='actor').to(device)  # 策略网络【神经网络】模型
        self.critic = Model(config, mode='critic').to(device)  # 价值网络模型

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)  # 策略网络的优化器
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)  # 价值网络的优化器

        self.epochs = config.epochs
        self.horizon = config.horizon  # 视野范围，应考虑多远的未来奖励
        self.time_steps = config.time_steps  # 时间步长，决策的步数
        self.lr = config.lr
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # 梯度累积步骤，梯度更新之前，进行几次前向和反向传播
        self.gamma = config.gamma  # 折扣因子
        self.clip = config.clip  # 裁剪值

        self.model_path = config.model_path  # 模型保存路径
        self.input_record = {'input': [], 'output': [], 'record': []}  # 记录输入和输出

        # 如果存在模型，加载 actor 模型
        if os.path.exists(f'./{self.model_path}/state_dict.pt'):
            self.actor.load_state_dict(torch.load(f'./{self.model_path}/state_dict.pt'))

    # 训练
    def train(self):
        # 加载日志
        if os.path.exists(f'{self.model_path}/logger.pkl'):
            with open(f'{self.model_path}/logger.pkl', 'rb') as file:
                logger = pickle.load(file)
        else:
            logger = {
                'actor_loss': [],
                'critic_loss': [],
                'pretrain_loss': [],
                'max_reward': [],
                'min_reward': [],
                'avg_reward': []
            }
        # 加载历史记录
        if os.path.exists(f'{self.model_path}/history.pkl'):
            with open(f'{self.model_path}/history.pkl', 'rb') as file:
                history = pickle.load(file)
        else:
            history = {
                'observations': [],  # 观察到的状态
                'actions': [],  # 采取的行动
                'log_probs': [],  # 对数概率
                'advantages': []  # 优势估计
            }

        # 遍历每个 epoch
        for e in range(self.epochs):
            print('epochs: ', e)

            # 1. 当前策略采样一次轨迹，得到所有时间步的观察、动作、奖赏
            batch_observations, batch_actions, batch_log_probs, batch_advantages, batch_rewards = self.rollout()
            rewards = batch_rewards.sum(1)
            print(rewards)
            # 更新该 epoch 的历史记录
            history['observations'].append(batch_observations)
            history['observations'] = history['observations'][-self.horizon:]  # 限制视野 horizon，取最后 self.horizon 个元素
            history['actions'].append(batch_actions)
            history['actions'] = history['actions'][-self.horizon:]
            history['log_probs'].append(batch_log_probs)
            history['log_probs'] = history['log_probs'][-self.horizon:]
            history['advantages'].append(batch_advantages)
            history['advantages'] = history['advantages'][-self.horizon:]

            # 2. 遍历所有 epochs 的历史数据
            for idx, sample in enumerate(
                    zip(history['observations'], history['actions'], history['log_probs'], history['advantages'])):
                print('history: ', idx)
                batch_observations, batch_actions, batch_log_probs, batch_advantages = sample

                # 2.1 价值网络模型（Critic）得到动作价值
                batch_values = self.evaluate(batch_observations, batch_actions)[0]
                curr_values, curr_log_probs = self.evaluate(batch_observations, batch_actions)

                # 2.2 计算优势函数 = 当前策略的预期回报 - 当前状态的价值估计
                advantage = batch_advantages - batch_values.detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)  # 标准化处理

                # 2.3 计算损失
                ratios = torch.exp(curr_log_probs - batch_log_probs)  # 计算动作概率比率
                l1 = ratios * advantage  # TRPO 目标
                l2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage  # PPO 截断目标
                l3 = self.PreTrainLoss()  # 预训练【minimax】损失
                # actor 损失
                actor_loss = (-torch.min(l1, l2)).mean() + l3

                # critic 损失
                critic_loss = nn.MSELoss()(curr_values, batch_advantages)  # 均方误差

                # 2.4 向后传播
                actor_loss.backward()
                critic_loss.backward()

                # 2.5 优化【更新参数】
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    self.actor_optim.step()
                    self.actor_optim.zero_grad()

                    self.critic_optim.step()
                    self.critic_optim.zero_grad()

                # 记录日志
                logger['actor_loss'].append(actor_loss.item())
                logger['critic_loss'].append(critic_loss.item())
                logger['pretrain_loss'].append(l3.item())

            # 每个棋盘【进程】，所有时间步的奖赏 [time_step, 32] -> [, 32]
            # rewards = batch_rewards.sum(1)
            logger['max_reward'].append(rewards.max().item())
            logger['min_reward'].append(rewards.min().item())
            logger['avg_reward'].append(rewards.mean().item())
            print('avg_reward: ', rewards.mean().item())

            # 保存日志和历史记录
            self.save(logger, history)

        return logger

    # 2.1 价值网络模型（Critic）评价
    def evaluate(self, batch_observations, batch_actions):
        """
        Returns:
            batch_values：存储【价值】
            batch_log_probs：存储【对数概率】
        """
        batch_values, batch_log_probs = [], []

        # 遍历轨迹
        for observations, actions in zip(batch_observations, batch_actions):
            # 使用策略网络模型（Critic），得到动作评价矩阵 [32, 1]
            batch_values.append(self.critic(**observations))
            # 得到对数概率
            batch_log_probs.append(sample_from_dist(self.actor(**observations), actions)[1].unsqueeze(1))

        # [32, time_step]
        batch_values = torch.cat(batch_values, dim=1).to(device)
        # [32, time_step]
        batch_log_probs = torch.cat(batch_log_probs, dim=1).to(device)

        return batch_values, batch_log_probs

    # 1. 策略网络模型（actor）采样一次轨迹
    def rollout(self):
        """
        Returns:
            batch_observations：存储每个时间步的【观察结果】
            batch_actions：存储每个时间步的【动作矩阵】
            batch_log_probs：存储【对数概率】
            batch_advantages：存储【回报】
            batch_rewards：存储每个时间步的【奖励】[time_step, 32]
        """
        # 初始化
        batch_observations, batch_actions, batch_log_probs, batch_rewards = [], [], [], []
        # 重置环境，开始新的游戏回合 epoch
        observations, boards = self.environment.reset()

        # 移动到设备
        for key in observations:
            if isinstance(observations[key], torch.Tensor):
                observations[key] = observations[key].to(device)
            elif isinstance(observations[key], dict):
                for subkey in observations[key]:
                    if isinstance(observations[key][subkey], torch.Tensor):
                        observations[key][subkey] = observations[key][subkey].to(device)

        # 遍历时间步
        for t in range(self.time_steps):
            print('time step: ', t)

            # 记录游戏状态
            # 遍历【每一个线程】的游戏状态
            for i in range(observations['state'].shape[0]):
                # 如果状态未记录过
                if tuple(observations['state'][i].tolist()) not in self.input_record['record']:
                    # list 形式的状态记录，用于判断是否记录
                    self.input_record['record'].append(tuple(observations['state'][i].tolist()))
                    # 游戏状态记录
                    self.input_record['input'].append(
                        {
                            'state': observations['state'][i].reshape(1, -1),
                            'action': {
                                'old_row': observations['action']['old_row'][i].reshape(1, -1),
                                'old_col': observations['action']['old_col'][i].reshape(1, -1),
                                'row': observations['action']['row'][i].reshape(1, -1),
                                'col': observations['action']['col'][i].reshape(1, -1),
                            }
                        }
                    )
                    # 对于当前棋盘状态的最优行动
                    self.input_record['output'].append(torch.LongTensor([minimax(boards[i], 2)[2]]))

            # 收集【观察结果】
            batch_observations.append(observations)

            # 环境前进一步【黑棋、白棋分别走一步】
            actions, log_prob = self.get_action(observations)  # 使用策略网络模型（actor）得到动作矩阵 [32, 256]
            observations, rewards, done, boards = self.environment.step(actions)

            # 移动到设备
            for key in observations:
                if isinstance(observations[key], torch.Tensor):
                    observations[key] = observations[key].to(device)
                elif isinstance(observations[key], dict):
                    for subkey in observations[key]:
                        if isinstance(observations[key][subkey], torch.Tensor):
                            observations[key][subkey] = observations[key][subkey].to(device)
            rewards = rewards.to(device)

            # 收集【奖励】[, 32]
            batch_rewards.append(rewards.unsqueeze(1))
            # 收集【动作矩阵】
            batch_actions.append(actions)
            # 收集【对数概率】[, 32]
            batch_log_probs.append(log_prob.unsqueeze(1))

            # 【所有线程】的游戏都结束
            if sum(done) == len(done):
                break

        # 所有时间步的【奖励】[time_step, 32]
        batch_rewards = torch.cat(batch_rewards, dim=1)
        # 所有时间步的【回报】[time_step, 32]
        batch_advantages = self.compute_advantages(batch_rewards)
        # 所有时间步的【对数概率】[time_step, 32]
        batch_log_probs = torch.cat(batch_log_probs, dim=1)

        return batch_observations, batch_actions, batch_log_probs, batch_advantages, batch_rewards

    # 计算一批奖励（batch_rewards）的回报
    def compute_advantages(self, batch_rewards):
        batch_advantages = []

        # 遍历每个时间步的奖赏 [, 32]
        for rewards in batch_rewards:
            rewards = rewards.tolist()
            discounted_reward = 0
            advantages = []

            # 遍历每个【线程】的奖赏
            for reward in reversed(rewards):
                # 折扣奖赏 = 回报
                discounted_reward = reward + discounted_reward * self.gamma
                # 折扣奖赏列表 [, 32]
                advantages.insert(0, discounted_reward)

            # 所有时间步的折扣奖赏列表 [time_step, 32]
            batch_advantages.append(advantages)

        return torch.Tensor(batch_advantages).to(device)

    # 使用策略网络模型（actor）得到动作矩阵
    def get_action(self, obs):
        # actor 模型输出
        mean = self.actor(**obs)
        # 再采样动作、计算对数概率
        action, log_prob = sample_from_dist(mean)

        return action.detach(), log_prob.detach()

    # 2.3 计算预训练【minimax】损失
    def PreTrainLoss(self):
        loss = torch.Tensor([0]).to(device)
        # 遍历输入、输出
        for X, Y in zip(self.input_record['input'], self.input_record['output']):
            Y = Y.to(device)
            # 计算交叉熵损失 = 当前 actor 预测 - minimax 预测【作为最佳选择】
            loss += nn.CrossEntropyLoss()(self.actor(**X), Y)

        return loss / len(self.input_record['input'])

    def save(self, logger, history):
        torch.save(self.actor.state_dict(), f'{self.model_path}/state_dict.pt')
        with open(f'{self.model_path}/logger.txt', encoding='utf-8', mode='w') as file:
            file.write(str(logger))
        with open(f'{self.model_path}/logger.pkl', 'wb') as file:
            pickle.dump(logger, file)
        with open(f'{self.model_path}/history.pkl', 'wb') as file:
            pickle.dump(history, file)
        # with open(f'{self.model_path}/input_record.pkl', 'wb') as file:
        #     pickle.dump(input_record, file)