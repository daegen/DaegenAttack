import os
import torch
import json
# import utils
import torchvision.models as models
import torchvision.transforms as trans

from PIL import Image

__OUTPUT_DATA = './outputs/imagenet/'
__GEN_IMAGES = __OUTPUT_DATA + 'gen_images/'
__STATS = __OUTPUT_DATA + 'status/'

if not os.path.exists(__GEN_IMAGES):
    os.makedirs(__GEN_IMAGES)
if not os.path.exists(__STATS):
    os.makedirs(__STATS)

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')

IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')

__IMAGENET_MODELS = ['densenet121', 'googlenet', 'mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'shufflenet_v2_x1_0', 'squeezenet1_1','vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

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
        return self.norm(x.clone())

    def predict(self, x):
        norm_x = self.normalization(x.clone())
        predictions = self.model(norm_x).cpu()
        predictions = torch.softmax(predictions, 1)
        prob, label = torch.max(predictions, 1)
        return prob.item(), label.item()
        
class Pipeline:
    def __init__(self, models):
        self.models = tuple(models)
    
    def get_model(self, modelname):
        model = getattr(models, modelname)(pretrained=True)
        return ModelWrapper(model)
    
    def pipeline_prediction(self, x, discretesize=False):
        for modelname in self.models:
            model = self.get_model(modelname)
            with torch.no_grad():
                if discretesize:
                    yield (modelname, model.discrete_predict(x)[1])
                    continue
                yield (modelname, model.predict(x)[1])

def check_img(x_img):
    assert x_img.min() >= 0
    assert x_img.max() <= 255
    pass

def save_stats(modelnames, image_id, x_ori, x_adv, queries):
    """
        Expects images (x_ori, and x_adv) with pixel values between 0 and 255
        """
    check_img(x_ori)
    check_img(x_adv)
    
    pipeline = Pipeline(modelnames)

    stats = {"image": image_id, 
             'queries': queries, 
             'l2_norm': torch.norm((x_adv-x_ori)/255).item()}
    
    # pipeline classification for the original image
    for modelname, prediction in list(pipeline.pipeline_prediction(x_ori.clone())):
        key = 'original_prediction_%s' % (modelname)
        pred_stats = {key: prediction }
        stats.update(pred_stats)

    # pipeline classification for the adversarial image
    for modelname, prediction in list(pipeline.pipeline_prediction(x_adv.clone())):
        key = 'adversarial_prediction_%s' % (modelname)
        pred_stats = {key: prediction }
        stats.update(pred_stats)
    
    with open(__STATS + '%d_%s_%s_stats.json' %(image_id, modelnames[0], modelnames[1]), 'w') as fp:
        json.dump(stats, fp)
        fp.flush()
        fp.close()
            
    filename = '%d_%s_%s_adversary.jpg'%(image_id, modelnames[0], modelnames[1])
    
    new_im = Image.fromarray((x_adv.detach().cpu().numpy().transpose(1, 2, 0)).round().astype('uint8'))
    new_im.save(open(__GEN_IMAGES + filename, 'w'))
    
    new_im = Image.fromarray((x_ori.detach().cpu().numpy().transpose(1, 2, 0)).round().astype('uint8'))
    new_im.save(open(__GEN_IMAGES + str(image_id) + '_original.jpg', 'w'))