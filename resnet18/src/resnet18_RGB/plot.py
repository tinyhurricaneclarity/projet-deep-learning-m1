#affichage des résultats et des paramètres
import matplotlib.pyplot as plt

config_str = f"config{config_counter}, optimizer{optimizer_config}, epochs{num_epochs}, lr{learning_rate}, batch_size{batch_size}, momentum{momentum}, weight_decay{weight_decay}, step_size{step_size}, gamma{gamma}"
print(f"Running config {config_str}")


plt.figure(figsize=(12,5))

plt.suptitle(config_str, fontsize=8)
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Resnet18 - Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Resnet18 - Accuracy')
plt.legend()

plt.show()