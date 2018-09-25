from model import Net
import argparse
import torch
import numpy as np
import json

parser = argparse.ArgumentParser(description='Predict flower name from an image')
parser.add_argument('image_path', help='path to the image to be predicted')
parser.add_argument('checkpoint', help='location of the checkpointed model')
parser.add_argument('--top_k', type=int,
                    help='return top k most likely classes')
parser.add_argument('--category_names',
                    help='JSON file for mapping categories to real names')
parser.add_argument('--gpu', action='store_true',
                    help='use GPU (if available) for inference')
args = parser.parse_args()

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f) 

def reload(filepath):
    state = torch.load(filepath)
    model_state_dict = state['model_state_dict']
    model = state['model']
    
    return (model, state['class_to_idx'])

model, mapping = reload(args.checkpoint)

if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

model.to(device)
model.eval()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    image = Image.open(image)
    w, h = image.size
    aspect_ratio = h/w
    if w>h:
        new_h = 256
        new_w = int(256/aspect_ratio)
    else:
        new_w = 256
        new_h = int(256*aspect_ratio)

    image = image.resize((new_w, new_h))
    left = int((new_w - 224)/2)
    upper = int((new_h - 224)/2)
    right = left+224
    lower = upper+224
    cropped = image.crop((left, upper, right, lower))
    
    np_image = np.array(cropped)/255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    normalized = (np_image-means)/std
    return torch.from_numpy(normalized.transpose((2, 0, 1)))

def predict(image_path, model, topk=5, mapping=mapping):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.set_grad_enabled(False):
        image = process_image(image_path)
        image = image.unsqueeze(0).float()
        image = image.to(device)
        model.to(device)
        outputs = model(image)
        probs = outputs.exp()/outputs.exp().sum()

        top_k_prob, top_k_classes = torch.topk(probs, topk)
    
    top_k_prob = top_k_prob.squeeze().cpu().numpy().tolist()
    top_k_classes = top_k_classes.squeeze().cpu().numpy().tolist()
    
    inv_mapping = {v:k for k, v in mapping.items()}
    
    if type(top_k_classes)!=int:
        top_k_classes = [inv_mapping[k] for k in top_k_classes]
    else:
        top_k_classes = inv_mapping[top_k_classes]

    return (top_k_prob, top_k_classes)

probs, classes = predict(args.image_path, model, topk = args.top_k if args.top_k is not None else 1)

if type(probs)==float:
    probs = [probs]
    classes = [classes]

print('\n')
print('Prediction for the image: {}'.format(args.image_path))
print('\n')
print('-'*50)
print('{:<10}{:<15}{:<25}'.format('Class', 'Probability', 'Category Name'))
for p, c in zip(probs, classes):
    print('{:<10}{:<15.3f}{:<25}'.format(c, (p), cat_to_name[c] if args.category_names else 'NA'))
print('-'*50)

    