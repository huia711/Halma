# import json
#
#
# # 初始化参数
# class Config:
#     def __init__(self,
#                  num_processes: int = 32,
#                  time_steps: int = 50,
#                  epochs: int = 10000,
#                  dim: int = 32,
#                  n_head: int = 4,
#                  num_layers: int = 2,
#                  hidden_size: int = 64,
#                  dropout: float = 0.1,
#                  max_output: int = 256,
#                  lr: float = 0.005,
#                  gradient_accumulation_steps: int = 4,
#                  gamma: float = 0.95,
#                  clip: float = 0.2,
#                  horizon: int = 1000,
#                  model_path: str = './model2',
#                  log_path: str = './train_log'
#     ):
#         self.num_processes = num_processes
#         self.time_steps = time_steps
#         self.epochs = epochs
#         self.dim = dim
#         self.num_layers = num_layers
#         self.n_head = n_head
#         self.hidden_size = hidden_size
#         self.dropout = dropout
#         self.max_output = max_output
#         self.lr = lr
#         self.gradient_accumulation_steps = gradient_accumulation_steps
#         self.gamma = gamma
#         self.clip = clip
#         self.horizon = horizon
#         self.model_path = model_path
#         self.log_path = log_path
#
#     @staticmethod
#     def get_config(s):
#         config = json.loads(s)
#         return Config(**config)
