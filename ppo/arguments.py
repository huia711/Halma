import argparse


# 参数
def Config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_processes', type=int, default=32)  # 进程数，默认为32
    parser.add_argument('--time_steps', type=int, default=150)  # 时间步长
    parser.add_argument('--epochs', type=int, default=10000)  # 训练的轮数，默认为10000
    parser.add_argument('--dim', type=int, default=32)  # 维度，默认为32
    parser.add_argument('--n_head', type=int, default=4)  # 头的数量，用于多头注意力机制，默认为4
    parser.add_argument('--num_layers', type=int, default=2)  # 网络层数，默认为2
    parser.add_argument('--hidden_size', type=int, default=64)  # 隐藏层的大小，默认为64
    parser.add_argument('--dropout', type=float, default=0.1)  # dropout比率，默认为0.1
    parser.add_argument('--max_output', type=int, default=256)  # 最大输出，默认为256
    parser.add_argument('--lr', type=float, default=1e-3)  # 学习率，默认为0.001
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)  # 梯度累积步骤，默认为1
    parser.add_argument('--gamma', type=float, default=0.95)  # 折扣因子，默认为0.95
    parser.add_argument('--clip', type=float, default=0.2)  # 裁剪值，默认为0.2
    parser.add_argument('--horizon', type=int, default=1000)  # 视野范围【看多少历史】，默认为1000
    parser.add_argument('--model_path', type=str, default='./model')  # 模型保存路径，默认为'./model'

    return parser.parse_args()

