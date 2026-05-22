#Pour avoir la meme config pour tout le monde

class_names = ["Health", "Rust", "Other"]

config_counter = 1
num_epochs = 100
learning_rate = 0.1
batch_size = 32
momentum = 0.9
#weight_decay = 5e-4
step_size = 30
gamma = 0.1
criterion_config = "CrossEntropy"
optimizer_config = "SGD"
scheduler_config =  "StepLR"

config_dico = {
    "config_counter": config_counter,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "momentum": momentum,
    #"weight_decay": weight_decay,
    "step_size": step_size,
    "gamma": gamma,
    "criterion": criterion_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
  
}