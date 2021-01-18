import torch

gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 10
log_interval = 10
update_target = 10000 # 1000 for 5x5
replay_memory_capacity = 10000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

sequence_length = 32
burn_in_length = 4
eta = 0.9
local_mini_batch = 8
n_step = 2
over_lapping_length = 16