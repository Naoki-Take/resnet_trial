from models import get_model, get_optim, train_one_epoch, validate_one_epoch
from data import get_loader
from torchvision import models
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
parser = argparse.ArgumentParser(description="ResNet trial.")
parser.add_argument('--model', type=str,
                    default='resnet50', help='select model')
parser.add_argument('--num_classes', type=int,
                    default=10, help='number of classes')
parser.add_argument('--num_epochs', type=int,
                    default=3, help='number of epochs')
parser.add_argument('--optimizer', type=str, default="adam", help='optimizer')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dataset', type=str, default='mnist', help='dataset')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
args = parser.parse_args()


def main():
    print(f"GPU availability: {torch.cuda.is_available()}")
    model = get_model(args.model, args.num_classes)
    if args.use_gpu:
        model.to("cuda")
    optimizer = get_optim(args.model, args.num_classes, args.optimizer, args.lr)
    # optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_dataloader, val_dataloader = get_loader(args.dataset)
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.num_epochs):
        print(f"EPOCH {epoch+1}:")

        avg_loss, train_total_accuracy, train_label_accuracy = train_one_epoch(
            train_dataloader, model, loss_fn, optimizer)
        print(f"Train Loss: {avg_loss}, Train Total Accuracy: \
              {train_total_accuracy}, Train Label Accuracy: {train_label_accuracy}")
        # wandb.log({"train_loss": avg_loss, "train_total_accuracy": train_total_accuracy, **{f"train_acc_label_{label}": acc for label, acc in train_label_accuracy.items()}})

        val_loss, val_total_accuracy, val_label_accuracy = validate_one_epoch(
            val_dataloader, model, loss_fn)
        print(f"Validation Loss: {val_loss}, Validation Total Accuracy: \
              {val_total_accuracy}, Validation Label Accuracy: {val_label_accuracy}")
        # wandb.log({"val_loss": val_loss, "val_total_accuracy": val_total_accuracy, **{f"val_acc_label_{label}": acc for label, acc in val_label_accuracy.items()}})


if __name__ == "__main__":
    main()
