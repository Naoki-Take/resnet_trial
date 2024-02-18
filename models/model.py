import timm
import torch
import torch.nn as nn


def get_model(args.model, args.num_classes):
    if args.model == "resnet50":
        model = timm.create_model(
            'resnet50', pretrained=True, num_classes=args.num_classes)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    else:
        raise NotImplementedError
    return model


def get_optim(args.optimizer):
    parameters = get_model(args.model, args.num_classes).parameters()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer
