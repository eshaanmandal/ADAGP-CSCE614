import torchvision.models as models


def get_model(model_name, pretrained=False):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Model {model_name} is not supported')
    
    return model

    

