# ref: https://github.com/pytorch/examples/blob/master/mnist/main.py

import sys
import os

sys.path.append(os.environ['ARSENAL_PATH'])

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb

from torch.optim.lr_scheduler import StepLR

from pytorch_dnn_arsenal.dataset import build_dataset, DatasetSetting
from pytorch_dnn_arsenal.model import build_model, ModelSetting
from pytorch_dnn_arsenal.optimizer import build_optimizer, OptimizerSetting
from pytorch_dnn_arsenal.scheduler import build_scheduler, SchedulerSetting
from pytorch_dnn_arsenal.loss import build_criterion
from pytorch_dnn_arsenal.scheduler.damping_scheduler import DampingSchedulerSetting, build_damping_scheduler


def train(args, model, criterion, device, train_loader, optimizer, scheduler, damping_scheduler, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        if optimizer == 'kfac' and optimizer.steps % optimizer.TCov == 0:

            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),1).squeeze().cuda()
            loss_sample = criterion(output, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.

        optimizer.step()
        scheduler.step()
        damping_scheduler.step()

        # For Debug (LR Scheduler)
        print(f'lr: {scheduler.get_last_lr()[0]}' )
        print(f'damping: {damping_scheduler.get_last_damping()[0]}' )

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({
                'epoch':epoch,
                'train_loss':loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'damping':damping_scheduler.get_last_damping()[0]
            })
            
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--damping_coeficient_gamma',
                        type=float, default=0.999)
    parser.add_argument('--optimizer-name', type=str, default='kfac')
    parser.add_argument('--beta-update-rule', type=str, default='FR')                        
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = 'resnet8'
    model = build_model(
        ModelSetting(name=model_name, num_classes=10, dropout_ratio=0.5)
    ).to(device)
    print(model)

    criterion = build_criterion()
    
    dataset_name = 'cifar10'
    dataset = build_dataset(
        DatasetSetting(name=dataset_name, root='../example')
    )

    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset.val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset,  batch_size=args.batch_size, shuffle=False)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    
    optimizer = build_optimizer(
        OptimizerSetting(name=args.optimizer_name,
                         lr=args.lr,
                         weight_decay=0.005,
                         beta_update_rule=args.beta_update_rule,
                         model=model))

    scheduler = build_scheduler(
        SchedulerSetting(name = 'constant',
                        optimizer = optimizer,
                        enable_warmup = False,
                        max_epoch = args.epochs,
                        max_iteration = int(args.epochs*len(dataset.train_dataset)/args.batch_size),
                        warmup_multiplier = 1,
                        warmup_iteration = 0,
                        h_params = {})
    )

    damping_scheduler = build_damping_scheduler(
        DampingSchedulerSetting(name = 'constant',
                                optimizer = optimizer,
                                gamma=args.damping_coeficient_gamma,
                                step_size = 100000,
                                max_iteration = 100000000
        )

    )
    # wandb init
    print('wandb init')
    wandb_configs = {
        'dataset': dataset_name,
        'model': model_name,
        'optimizer': args.optimizer_name,
        'lr': args.lr
    }

    wandb_project_name = 'arsenal_exmaple'
    wandb.init(
        config=wandb_configs, 
        project=wandb_project_name, 
        entity='mlhpc',
        name=f'lr={args.lr}_optimizer={args.optimizer_name}_beta_update_rule={args.beta_update_rule}'
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, scheduler, damping_scheduler, epoch)

if __name__ == '__main__':
    main()
    
