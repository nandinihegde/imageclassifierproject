# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import torch
# from torch import nn
# from torch import tensor
# from torch import optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchvision import datasets, transforms
# import torchvision.models as models
import argparse

import utilityfuncs
import modelling

ap = argparse.ArgumentParser(description='Training a Image classification model')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpointsave/checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

print("These are the params you have set:")
print(ap.parse_args())
pa = ap.parse_args()
datapath= pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
architecture = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


image_datasets,dataloaders = utilityfuncs.load_data(datapath)


model, criterion, optimizer,classifier= modelling.classifier_setup(architecture,dropout,hidden_layer1,lr,power)


modelgrad,optimizergrad=modelling.train_model(model, optimizer, criterion, epochs, dataloaders, power)


modelling.save_checkpoint(image_datasets[0].class_to_idx,path,architecture,classifier,lr,epochs,optimizergrad,modelgrad)


print("Model has finished training!")