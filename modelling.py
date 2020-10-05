#import libraries
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import models
from PIL import Image
from torch.autograd import Variable
import numpy as np
#modelling the data

def classifier_setup(architecture,dropout,hidden_layer1,lr,power):
    model = getattr(models,architecture)(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #set classifer
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_layer1)),
                              ('relu', nn.ReLU()),
                              ('drop1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_layer1, 102)),
                              ('relu', nn.ReLU()),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    #Using Negative Log likelihood as out loss function
    criterion = nn.NLLLoss()
    #setting a classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    if torch.cuda.is_available() and power == 'gpu':
            model.cuda()

    return model, criterion, optimizer,classifier

#Insert validation function to perform validation
def validation(model, testloader, criterion,power):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        
        if torch.cuda.is_available() and power=='gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        else:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_model(model, optimizer, criterion, epochs, dataloaders, power):

    print_every = 100
    steps = 0
    #Train model on trainloaded
    # change to cuda or cpu based on availability
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda')
    else:
        model.to('cpu')
    #for i in keep_awake(range(5)):
    model=model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1
            if torch.cuda.is_available() and power=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')


            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloaders[1], criterion,power)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders[1])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))

                running_loss = 0

                # Make sure training is back on
                model.train()
    return model,optimizer

def save_checkpoint(class_toidx,path,architecture,classifier,lr,epochs,optimizer,model):
    # TODO: Save the checkpoint 
    checkpointstats = {'input_size': 25088,
                  'output_size': 102,
                  'arch': architecture,
                  'learning_rate': lr,
                  'batch_size': 64,
                  'classifier' : classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_toidx}

    torch.save(checkpointstats, path)
    
    
#process single image 
def process_image(imagepath):
        
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(imagepath)  
    im = im.resize((256,256))
    cut = 0.5*(256-224)
    im = im.crop((cut,cut,cut+224,cut+224))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im=im.transpose(2,0,1)
    
    return im
    
def predict(image_path, model, number_of_outputs, power):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
     # process image for prediction

    image=process_image(image_path)

    # eval mode
    model.eval()
    if torch.cuda.is_available() and power=='gpu':
        # Move model parameters to the GPU
        model.cuda()
        image = torch.from_numpy(np.array([image])).float()  
        image = Variable(image)
        image = image.cuda()
    else:
        model.cpu()
        image = torch.from_numpy(np.array([image])).float()
        image = Variable(image)
        image = image.cpu()
  
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    prob = torch.topk(probabilities, number_of_outputs,sorted=True)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, number_of_outputs,sorted=True)[1].tolist()[0] # index
   
    idx = []
    for i in range(len(model.class_to_idx.items())):
        idx.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    labels = []
    for i in range(number_of_outputs):
        labels.append(idx[index[i]])

    return prob, labels
    
