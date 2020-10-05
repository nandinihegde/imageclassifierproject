#import library
from torchvision import datasets, transforms, models
import torch
from torch import optim

#loading and preprocessing data

def load_data(data_dir  = "./flowers" ):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_vali_transforms = transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=test_vali_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_vali_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)


    image_datasets = [train_datasets, validation_datasets, test_datasets]
    dataloaders = [trainloader, validationloader, testloader]
    
    
    return image_datasets,dataloaders

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename):
      
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer =optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model
