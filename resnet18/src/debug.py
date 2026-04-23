"""Confusion nombre d'images et nombre de batch"""
import resnet18.src.resnet18_RGB.data_load as data_load

batch_size = 32
# Import des données train provenant de data.py

path_train_rgb = "/net/cremi/leanguye/projet-deep-learning-m1/resnet/data/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/RGB/"
x_train, y_train = data_load.load_data_train(path_train_rgb)

#Convertion en tensor et trainloader
dataset = data_load.CustomImageDataset(x_train, y_train, transform=None)

#Dataloader
train_loader, val_loader, test_loader, train_size, val_size = data_load.create_dataloader(dataset, batch_size=batch_size)


print("len_train_loader", len(train_loader), "len_val_loader", len(val_loader), "train_size", train_size, "val_size", val_size) #14 batch train 2 batch val
print("len_x_train", len(x_train), "len_y_train", len(y_train))
print("len_dataset", len(dataset))

for x, y in train_loader:
    print(x.shape)
    break