import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from dataset import MaskBaseDataset
from dataset import MaskSplitByProfileDataset
from loss import create_criterion
from loss import FocalLoss
import madgrad
from tqdm import tqdm

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed) #def: 42
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
        
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskSplitbyProfile
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18
    
    # -- data_loader
    #train_csv = pd.read_csv("/opt/ml/input/data/train/train.csv")
    
    labels = []
    print("Get Labels from dataset...")
    for i in tqdm(range(len(dataset))) :
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
     # -- augmentation
    '''
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    '''
    
    
    '''
    stratifiedkfold = StratifiedKFold(n_splits = 5,random_state = 42, shuffle = True)
    folds = []
    
    
    
    
    
    print("Total img counts : ", len(labels))
    
    
    for fold_index, (train_idx, valid_idx) in tqdm(enumerate(stratifiedkfold.split(range(len(labels)), labels))) :
        folds.append({'train' : train_idx, 'valid' : valid_idx})
        
    print()
    print(f'[fold: {fold_index+1}, total fold: {len(folds)}]')
    print(len(train_idx), len(valid_idx))
    print(train_idx)
    print(valid_idx)
    for fold in folds :
        train_subset = Subset(dataset=dataset, indices=train_idx)
        valid_subset = Subset(dataset=dataset, indices=valid_idx)
        train_loader = DataLoader(dataset=train_subset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=use_cuda,
                                  drop_last=True,
                                 )
        val_loader = DataLoader(dataset=valid_subset,
                                  batch_size=args.valid_batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=use_cuda,
                                  drop_last=True,
                                 )
    '''
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)
    #torch.nn.DataParallel : https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    # -- loss & metric
    #criterion = create_criterion(args.criterion)  # default: cross_entropy

    df_label = pd.Series(labels)
    label_sorted = df_label.value_counts().sort_index()
    n_label = torch.Tensor(label_sorted.values)
    gamma = 2
    normed_weights = [1 - (gamma*x/sum(n_label)) for x in n_label]
    normed_weights = torch.FloatTensor(normed_weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=normed_weights)
    criterion = create_criterion(args.criterion)


    #optimizer = madgrad.MADGRAD(params : any, lr = 0.001, momentum = 0.9, weight_decay = 0, eps = 1e-06)
    try :
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    except AttributeError :
        opt_module = getattr(import_module("madgrad"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0
    )
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, eta_min=0.000005)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=args.gamma)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    #train starts
    for epoch in tqdm(range(args.epochs)):
        # train loop
        print()
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels) + criterion2(outs, labels)
            #loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

            labels = labels.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            train_f1 = f1_score(labels, preds, average='macro')

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.4%} || lr {current_lr} || "
                    f"F1_score {train_f1:4.4} "
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1_score", train_f1, epoch * len(train_loader) + idx)
                loss_value = 0
                matches = 0

        scheduler.step() #lr scheduler

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                loss_item2 = criterion2(outs, labels).item()
                loss_item = list(np.add(np.array(loss_item),np.array(loss_item2)))
                acc_item = (labels == preds).sum().item()

                labels = labels.cpu().detach().numpy()
                preds = preds.cpu().detach().numpy()
                f1_item = f1_score(labels, preds, average='macro')

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, preds, args.dataset != "MaskSplitByProfileDataset")

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)

            best_val_acc = max(best_val_acc, val_acc)

            if val_loss < best_val_loss:
                print(f"New best model for val_loss : {val_loss:4.4}! saving the best loss model..")
                torch.save(model.module.state_dict(), f"{save_dir}/{args.model}_epoch{epoch}_loss_{val_loss}.pth")
                best_val_loss = val_loss
            if val_f1 > best_val_f1:
                print(f"New best model for val_F1_score : {val_f1:4.4}! saving the best F1_score model..")
                torch.save(model.module.state_dict(), f"{save_dir}/{args.model}_epoch{epoch}_f1_{val_f1}.pth")
                best_val_f1 = val_f1
            print(
                f"[Val] loss: {val_loss:4.4}, F1_score {val_f1:4.4}, acc : {val_acc:4.4%} || "
                f"best loss: {best_val_loss:4.4}, best_F1_score {best_val_f1:4.4} , best acc : {best_val_acc:4.4%} "
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1_score", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=18, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset type (default: MaskBaseDataset)')
    #parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    #parser.add_argument("--resize", nargs="+", type=list, default=[300, 256], help='resize size for image when training (default 128 96)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Efficientnet_B3', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='MADGRAD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.00008, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=3, help='learning rate scheduler(stepLR) deacy step (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate scheduler(stepLR) gamma (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=16, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='experiments', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '../trained_models/Efficientnet_B3'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)