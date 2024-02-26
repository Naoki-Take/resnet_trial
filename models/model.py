import timm
import torch
import torch.nn as nn


def get_model(args_model, args_num_classes):
    if args_model == "resnet50":
        model = timm.create_model(
            'resnet50', pretrained=True, num_classes=args_num_classes)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args_num_classes)
    elif args_model == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, args_num_classes)
        params_to_update = []
        update_param_names = ['head.weight', 'head.bias']
        for name, param in model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError
    return model


def get_optim(args_model, args_num_classes, args_optimizer, args_lr):
    parameters = get_model(args_model, args_num_classes).parameters()
    if args_optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args_lr, momentum=0.9)
    if args_optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args_lr)
    else:
        raise NotImplementedError
    return optimizer
