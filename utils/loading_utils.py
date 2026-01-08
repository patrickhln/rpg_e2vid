import torch
from model.model import *


def _xpu_available():
    """Check if Intel XPU (via IPEX) is available."""
    try:
        import intel_extension_for_pytorch as ipex
        return torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False


def load_model(path_to_model):
    print('Loading model {}...'.format(path_to_model))
    if torch.cuda.is_available() or _xpu_available():
        raw_model = torch.load(path_to_model)
    else:
        raw_model = torch.load(path_to_model, map_location=torch.device('cpu'))
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif use_gpu and _xpu_available():
        device = torch.device('xpu:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
