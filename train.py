import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
import copy
from skimage import io 
import matplotlib.pyplot as plt 
from functools import reduce
from datetime import datetime
import os
import argparse
import warnings
import numpy as np
from model import Net
from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Train a Deep Learning Model')
parser.add_argument('data_directory',
	help='location of the data directory')
parser.add_argument('-sd', '--save_dir',
	help='location to save model checkpoints')
parser.add_argument('-a', '--arch',
	help='use pre-trained model',
	choices=['vgg16', 'resnet18', 'alexnet', 'squeezenet1_0', 'densenet161'],
	default='resnet18')
parser.add_argument('-lr', '--learning_rate',
	type=float, default=0.01, 
	help='set the learning rate (hyperparameter)')
parser.add_argument('-bs', '--batch_size',
	type=int, default=128,
	help='set the batch size (hyperparameter)')
parser.add_argument('-e', '--epochs',
	type=int, default=5,
	help='set the number of training epochs')
parser.add_argument('--gpu', action='store_true',
	help='flag to use GPU (if available) for training')
parser.add_argument('--hidden_units', type=int,
	default=512)
parser.add_argument('-nw', '--num_workers',
	type=int, default=4,
	help='set the number of workers')
args = parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

batch_size = args.batch_size
num_workers = args.num_workers

data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
def train_model(model, criterion, optimizer, num_epochs=25, dataset_sizes=dataset_sizes):
    start = datetime.now()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], 
                                       total=dataset_sizes[phase]//batch_size,
                                      desc=phase.upper()):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                #print(running_corrects.item(), dataset_sizes[phase],running_corrects.item()/dataset_sizes[phase])
                
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.upper(), epoch_loss, epoch_acc))
            
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
        
    time_elapsed = datetime.now() - start

    print('Training complete in '+str(time_elapsed))
    print('Best Valid Loss: {:.4f} Acc: {:.4f}'.format(best_loss, best_acc))

    model.load_state_dict(best_model_wts)
    return model        

model = Net(args.arch)
model.to(device)
criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters)

model = train_model(model, criterion, optimizer, num_epochs=args.epochs)

state = {'optimizer': optimizer,
             'criterion': criterion,
             'model_state_dict':model.state_dict(),
             'model':model,
             'class_to_idx': image_datasets['train'].class_to_idx}

if not args.save_dir:
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(state, 'checkpoints/model.pth')
    print('Model checkpointed at checkpoints/model.pth')
else:
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(state, args.save_dir+'/model.pth')
    print('Model checkpointed at {}/model.pth'.format(args.save_dir))


 
                 
        
