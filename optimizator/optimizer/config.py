
device = "cuda:0"
learning_rate = 1.0
nb_points = 100
iterations = 750
kept_threshold = 1e-4
merge_threshold = 1e-4

exponential_mean = 0.9
scheduler_factor = 0.1
scheduler_patience = 100
scheduler_min_lr = 1e-8
finetuning_iterations = 10

checkpoint_path = None
# checkpoint_path = './checkpoint/checkpoint.pt'