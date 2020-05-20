import argparse
import os
import json
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans

from torchvision.datasets import ImageFolder

import torchvision.models as models

from PIL import Image

from solvers.hc import HillClimbing
from save_stats import save_stats

# set manual seed
__SEED = 2809
torch.manual_seed(__SEED)
np.random.seed(__SEED)

__IMAGENET_MODELS = ['vgg16', 'vgg19', 'resnet34', 'mobilenet_v2', 'densenet121']

IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')

class Normalize(torch.nn.Module):
    """
    Simple Module to normalize an batch of images
    """

    def __init__(self, mean, std, scale=1.0):
        super(Normalize, self).__init__()
        mean = torch.tensor(mean).float()
        self.mean = mean

        std = torch.tensor(std).float()
        self.std = std
        
        self.scale = scale

    def forward(self, x):
        mean = self.mean.type_as(x)
        mean = mean[None, :, None, None]

        std = self.std.type_as(x)
        std = std[None, :, None, None]
        
        x /= self.scale
        
        return (x - mean) / std
    
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD, scale=255.0)
        
    def normalization(self, x):
        return self.norm(x)

    def predict(self, x):
        x = self.normalization(x.clone())
        prob, label = torch.max(torch.softmax(self.model(x), 1), 1)
        return prob.item(), label.item()
    
def main(args):
    first_model = ModelWrapper(getattr(models, args.first_model)(pretrained=True))
    second_model = ModelWrapper(getattr(models, args.second_model)(pretrained=True))
    dataset = ImageFolder(root='./datasets/Imagenet/Sample_1000/', transform=IMAGENET_TRANSFORM)
    for i in range(args.seeds):
        print(i)
        img_t, _ = dataset[i]
        img_t *= 255
        img_t = img_t.int()
        max_score = 1
        c = 0.0

        input_shape = img_t.shape

        def evaluation(x_t, x_adv):
            pred_a, label_a = first_model.predict(torch.from_numpy(x_adv).to(device).view(input_shape).unsqueeze(0))
            pred_b, label_b = second_model.predict(torch.from_numpy(x_adv).to(device).view(input_shape).unsqueeze(0))
            
            pred_part = abs(pred_a -  pred_b)
            norm_part = (c * np.linalg.norm(x_adv-x_t)/np.linalg.norm(x_adv))
            return pred_part - norm_part if label_a == label_b else max_score

        x_t = img_t.float().view(-1).cpu().numpy()

        start_time = time.perf_counter()
        hc = HillClimbing(X_t=x_t, permutation=False, max_score=max_score)
        hc.evaluation = evaluation
        
        solution_score, sol, queries = hc.solve(verbose=True)
        end_time = time.perf_counter()
        
        sol_t = torch.from_numpy(sol).float().view(input_shape)
        save_stats([args.first_model, args.second_model], i, img_t.float().to(device), sol_t.float().to(device), queries)
        
        print('Completed in: %d epochs, best score %.4f. Total time: %d ' %(queries, solution_score, end_time - start_time))
              
if __name__ == '__main__':
    description = 'Main function for difference-inducing input generation with Daegen'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-q', '--queries', default=10000, type=int, help="Query bugdet for each model during the attack")
    parser.add_argument('-s', '--seeds', default=100, type=int, help="Number of images to be used during the attack")
    parser.add_argument('-fm', '--first-model', default='vgg16', type=str, help="Name of the first the model", choices=__IMAGENET_MODELS)
    parser.add_argument('-sm', '--second-model', default='vgg19', type=str, help="Name of the first the model", choices=__IMAGENET_MODELS)

    args = parser.parse_args()
    main(args)