import numpy as np
from halma.board import Board
from minimax.algorithm import minimax, get_all_moves
from .arguments import Config
import torch


print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# 将棋盘 board 转换为 PyTorch 张量
# 黑子、白子和空位分别用 1、2 和 0 表示
def process(board):
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col] == 0:
                board[row][col] = 0
            elif board[row][col].color == WHITE:
                board[row][col] = 2
            else:
                board[row][col] = 1
    return torch.LongTensor(board.tolist()).unsqueeze(0)


# 对不等长的列表进行填充，确保它们具有相同的长度
def pad(X):
    ret = []
    max_len = max([len(x) for x in X])
    for x in X:
        ret.append(x + [0 for _ in range(max_len - len(x))])
    return ret


# 将游戏状态转换为 PyTorch 张量
def state(Boards):
    ret = []  # PyTorch 处理后的棋盘
    old_row, old_col, row, col = [], [], [], []  # 所有黑色棋子的有效走法

    # 遍历【每一个线程】的游戏
    for Board in Boards:
        # PyTorch 处理 game 的棋盘
        board = np.array(Board.board)[:8, :8]
        ret.append(process(board))

        # 处理 game 状态所有黑色棋子的有效走法
        orl, ocl, rl, cl = [], [], [], []
        for piece in Board.get_all_pieces(BLACK):
            for move in Board.get_valid_moves(piece):
                # 移动前
                orl.append(piece.row + 1)
                ocl.append(piece.col + 1)
                # 移动后
                rl.append(move[0] + 1)
                cl.append(move[1] + 1)
        old_row.append(orl)
        old_col.append(ocl)
        row.append(rl)
        col.append(cl)

    return {'state': torch.cat(ret, dim=0).reshape(-1, 64),
            'action': {'old_row': torch.LongTensor(pad(old_row)), 'old_col': torch.LongTensor(pad(old_col)),
                       'row': torch.LongTensor(pad(row)), 'col': torch.LongTensor(pad(col))}}


# PPO 环境
class HalmaGameEnvironment:
    def __init__(self, config: Config()):
        self.num_processes = config.num_processes  # 进程数：同时运行的游戏数量
        self.boards = []  # 棋盘 boards
        self.done = []  # 每个游戏是否结束

    # 重置环境，开始新的游戏回合
    def reset(self):
        self.boards = [Board() for _ in range(self.num_processes)]
        self.done = [0 for _ in range(self.num_processes)]

        return state(self.boards), self.boards

    # 环境前进一步【黑棋、白棋分别走一步】
    def step(self, actions):
        """
        action:
            shape [bs, nd]
            shape [批量大小, 动作数量]

        Return:
            state, rewards [, 32], done [, 32], boards
        """
        rewards = []

        # 遍历【每一个线程】的游戏
        for idx in range(len(self.boards)):
            # 棋盘和奖赏
            board = self.boards[idx]
            reward = 0

            # 如果游戏结束
            if self.done[idx]:
                # 更新奖赏 +0
                rewards.append(0)
                continue

            # 如果未结束
            # 返回棋盘上黑色的所有可能走法
            moves = get_all_moves(board, BLACK)
            if len(moves) > 0:
                # 使用 Softmax 函数选择一个动作
                action = torch.nn.functional.softmax(actions[idx, :len(moves)], dim=0).argmax(0).item()
                board = moves[action]
                # 更新奖赏 -eva
                reward += -board.evaluate()

            # 白棋使用 minimax 策略更新一步
            evaluate, board = minimax(board, 1, float('-inf'), float('inf'), True)
            reward += -evaluate

            # 判断是否结束（出现胜者）
            winner = board.winner()
            if winner == 'Black wins':
                reward += 1000
                self.done[idx] = 1  # 标记结束
            elif winner == 'White wins':
                reward -= 1000
                self.done[idx] = 1

            # 更新该【线程】奖赏
            rewards.append(reward)
            # 更新该【线程】棋盘
            self.boards[idx] = board

        return state(self.boards), torch.Tensor(rewards), self.done, self.boards
