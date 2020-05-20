# DaegenAttack

# Dependencies
- python >= 3.6
- PyTorch
- Torchvision
- Pillow

To install the dependencies:

```pip install torch torchvision numpy Pillow```

# Run Demo

### Algorithm options
- -fm to define the first model [vgg16|vgg19|resnet34|mobilenet_v2|densenet121]
- -sm to define the second model [vgg16|vgg19|resnet34|mobilenet_v2|densenet121]
- -s specify the number of seeds

### Running attack on Imagenet using Neural Networks VGG16 and VGG19 for 20 seeds images
```python attack.py -fm vgg16 -sm vgg19 -s 20```

