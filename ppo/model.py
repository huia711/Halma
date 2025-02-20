import torch
import torch.nn as nn
from .arguments import Config


# 状态嵌入层
class StateEmbedding(nn.Module):
    def __init__(self, dim):
        # 父类 nn.Module 的初始化方法
        super(StateEmbedding, self).__init__()
        # pytorch 嵌入层【空格、黑子、白子】
        self.embeddings = nn.Embedding(3, dim)

    # 前向传播
    def forward(self, X):
        # 输出[3, 32]
        return self.embeddings(X)


# 动作嵌入层
class ActionEmbedding(nn.Module):
    def __init__(self, dim):
        super(ActionEmbedding, self).__init__()
        # 嵌入层 【9x9 的棋盘的行和列】
        self.row_embeddings = nn.Embedding(9, dim, padding_idx=0)
        self.col_embeddings = nn.Embedding(9, dim, padding_idx=0)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    # 前向传播
    def forward(self, old_row, old_col, row, col):
        # 旧状态嵌入
        old_state = self.row_embeddings(old_row) + self.col_embeddings(old_col)
        # 新状态嵌入
        state = self.row_embeddings(row) + self.col_embeddings(col)
        # 全连接层【特征融合】
        # 输出[4*9, 32]
        return self.fc(torch.cat([old_state, state], dim=-1))


# 整体模型
class Model(nn.Module):
    def __init__(self, config: Config(), mode='actor'):  # Actor: 选择一个动作；Critic: 评价当前状态或动作
        super(Model, self).__init__()
        dim = config.dim  # 维度【与线程数相同】，默认为32
        n_head = config.n_head
        hidden_size = config.hidden_size  # 隐藏层的大小，默认为64
        num_layers = config.num_layers  # 网络层数，默认为2
        dropout = config.dropout
        max_output = config.max_output  # 动作数量，默认为256

        # 状态嵌入层
        self.se = StateEmbedding(dim)
        # 动作嵌入层
        self.ae = ActionEmbedding(dim)
        # transformer 单元
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_head,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation=nn.functional.gelu,
                batch_first=True
            ),
            num_layers=num_layers
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_output if mode == 'actor' else 1)    # Actor: 可选的不同动作的分数；Critic: 当前状态或动作的分数
        )

    # 前向传播
    def forward(self, state, action):
        # 状态嵌入层 [3, 32]
        state = self.se(state)
        # 动作嵌入层 [4*9, 32]
        action = self.ae(**action)

        # transformer 单元 [4*9+3, 32] --> [4*9+3, 32]
        X = torch.cat([state, action], dim=1)  # （每个【线程】的状态和动作）连接起来
        X = self.transformer(X)

        # FC 全连接层 [4*9+3, 32] -mean-> [1, 32] --> [256, 32]/[1, 32]
        X = X.mean(1)  # 平均，得到每个【线程】的状态均值
        return self.fc(X)
