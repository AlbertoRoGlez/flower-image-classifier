import nnlib, argparse

ap = argparse.ArgumentParser(description='Train.py')

# Cmd arguments
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=40)
ap.add_argument('--gpu', dest="gpu", action="store", default=True, type=bool)
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./trained_model.pth")

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
architecture = pa.arch
dropout = pa.dropout
hidden_layer = pa.hidden_units
gpu = pa.gpu
epochs = pa.epochs

print("========= START OF MODEL TRAINING. =========")

dataloaders, image_datasets = nnlib.load_data(where)
model, criterian, optimizer = nnlib.nn_builder(architecture,dropout,hidden_layer,lr,gpu)
nnlib.train_nn(model, criterian, optimizer, dataloaders, image_datasets, epochs, gpu)
nnlib.save_checkpoint(model, image_datasets, path, architecture, hidden_layer, dropout, lr, epochs)


print("========= MODEL TRAINED. =========")