import torch
from torchvision import datasets, models
import torchvision.transforms as transforms
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import time
import json
import copy
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

train_on_gpu = torch.cuda.is_available()

#TRAIN FUNCTION

def load_data(where = '../aipnd-project/flowers'):
    print('load data...')
    data_dir = '../aipnd-project/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }


    dirs = {
        'train': train_dir,
        'valid': valid_dir,
        'test' : test_dir
    }

    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes

    return dataloaders , class_names, dataset_sizes, image_datasets

def set_net(structure='vgg19',lr = 0.001, power='cuda'):
    print('set archi...')
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("Please choose models between vgg16 or vgg19")



    for param in model.features.parameters():
        param.requires_grad = False


    n_inputs = model.classifier[6].in_features

    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, 102)
    model.classifier[6] = last_layer

    if train_on_gpu:
        if power=='cuda':
            model.cuda()
        else:
            print('GPU have to be cuda compatible')
       


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr)

    return model, optimizer, criterion

def train_and_save(model, criterion, optimizer, loader, epochs = 1, power='cuda'):
    print('train...')

    # number of epochs to train the model
    epochs = epochs
    valid_loss_min = np.Inf

    for epoch in range(1, epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        for data, target in loader['train']:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                if power == 'cuda':
                    data, target = data.cuda(), target.cuda()
                else:
                    print('You need a cuda compatible CPU my friend')

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)        

        model.eval()
        for data, target in loader['valid']:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                if power == 'cuda':
                    data, target = data.cuda(), target.cuda()
                else:
                    print('your GPU is not cuda compatible')

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            
        train_loss = train_loss/len(loader['train'].dataset)
        valid_loss = valid_loss/len(loader['valid'].dataset)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

# PREDICT FUNCTIONS
            
def process_image(input_img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    img = Image.open(input_img)
    
    # Resize image
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        
    # Crop image
    bottom_margin = (img.height-224)/2
    top_margin = bottom_margin + 224
    left_margin = (img.width-224)/2
    right_margin = left_margin + 224
    
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize image
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    # move to first dimension --> PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image_processed, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image_processed.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def load_model(model, model_saved_name='model.pt'):
    model.load_state_dict(torch.load(model_saved_name))
    return model

def predict_image(image_datasets, model, cat_to_name, input_img, top_k=5, power='cpu'):
    print('predict...')
    img = process_image(input_img)
    #transform to tensor
    tensor_img = torch.from_numpy(img).type(torch.FloatTensor)
    #batch 1
    input_model = tensor_img.unsqueeze(0)

    if power == 'cpu':
        model = model.cpu()
    else:
        print("you have to set power='cpu' ")

    log_results = model(input_model)
    proba = torch.exp(log_results)

    top_proba, top_label = proba.topk(top_k)
    top_proba = top_proba.detach().numpy().tolist()[0] 
    top_label = top_label.detach().numpy().tolist()[0]

    # transform indices to classes name
    idx_to_class = {val: key for key, val in image_datasets['train'].class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_label]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_label]
    
    # plot
    #plt.figure(figsize = (6,10))
    #ax = plt.subplot(2,1,1)

    #with open(json_map_name, 'r') as f:
        #cat_to_name = json.load(f)
    
    # Set up title
    flower_num = input_img.split('/')[4]
    title_ = cat_to_name[flower_num]
    
    print(f"top 5 flowers and probabilities respectively {top_flowers} --> {top_proba}")

    #UNCOMMENT TO PLOT 
    
    # Plot flower
    #img = process_image(image_path)
    #imshow(img, ax, title = title_);
    
    # Plot bar chart
    #plt.subplot(2,1,2)
    #sns.barplot(x=top_proba, y=top_flowers, color=sns.color_palette()[0]);
    #plt.show()            
            
            
            
            
            