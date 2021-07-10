import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def local_iteration(loader,model,optimizer,criterion,device,steps=1):
    i = 0
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        predicts = model(inputs)
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()
        i+=1
        if i==steps:
            break
    return None

def worker_operations(model,ps_model_flat,error,ps_model_mask,device,args):
    model_flat = sf.get_model_flattened(model, device)
    model_flat.add_(error)

    difmodel = (model_flat.sub(ps_model_flat)).to(device)
    difmodel_clone = torch.clone(difmodel.detach()).to(device)


    sf.sparse_timeC(difmodel, args.sparsity_window, 10, ps_model_mask, device)

    if args.quantization:
        difmodel = sf.quantize(difmodel, args, device)
    new_error = (difmodel_clone.sub(difmodel))

    return difmodel,new_error




def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)


    net_users = [get_net(args).to(device) for u in range(num_client)]

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay = 1e-4) for cl in range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=[150,225], gamma=0.1) for cl in range(num_client)]


    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = 50000
    modelsize = sf.count_parameters(net_ps)
    errors = []
    accuracys = []
    bias_mask = sf.get_BN_mask(net_ps, device)
    ps_model_mask = torch.ones(modelsize).to(device)
    currentLR = sf.get_LR(optimizers[0])
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s/(args.bs * num_client * args.LSGDturn))
    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                             shuffle=True) for cl in range(num_client)]
    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = args.warmUp and epoch <5
        if atWarmup:
            sf.lr_warm_up(optimizers, epoch ,args.lr)
            warm_runs = int(runs * args.LSGDturn)
            for run in range(warm_runs):
                for cl in range(num_client):
                    local_iteration(trainloaders[cl], net_users[cl], optimizers[cl], criterions[cl], device)
                sf.initialize_zero(net_ps)
                [sf.push_model(net,net_ps,num_client) for net in net_users]

        else:
            for run in range(runs):
                for cl in range(num_client):
                    local_iteration(trainloaders[cl],net_users[cl],optimizers[cl],criterions[cl],device,steps=args.LSGDturn)

                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat)

                for cl in range(num_client):
                    difmodel, new_error = worker_operations(net_users[cl],ps_model_flat,errors[cl],ps_model_mask,device,args)
                    ps_model_dif.add_(difmodel/num_client)
                    errors[cl] = new_error

                ##server-side###
                ps_model_flat.add_(ps_model_dif)
                topk = math.ceil(ps_model_dif.nelement() / args.sparsity_window)
                ind = torch.topk(ps_model_dif.abs(), k=topk, dim=0)[1]
                ps_model_mask *= 0
                ps_model_mask[ind] = 1
                if args.biasFairness:
                    ps_model_mask.add_(bias_mask)
                    ps_model_mask = (ps_model_mask>0).int()
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                #################

        acc = evaluate_accuracy(net_ps, testloader, device)
        [schedulers[cl].step() for cl in range(num_client)]
        lr_ = sf.get_LR(optimizers[0])
        if lr_ != currentLR and not atWarmup:
            [error.mul_(lr_ / currentLR) for error in errors]
        currentLR = lr_
        accuracys.append(acc * 100)
        print('accuracy:',acc*100,)
    return accuracys

def train_topk(args,device):
    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)

    net_users = [get_net(args).to(device) for u in range(num_client)]

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True)
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=[150, 225], gamma=0.1) for cl in
                  range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = 50000
    modelsize = sf.count_parameters(net_ps)
    errors = []
    accuracys = []
    currentLR = sf.get_LR(optimizers[0])
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s / (args.bs * num_client))
    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    k = math.ceil(modelsize/args.sparsity_window)
    trainloaders = [DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                               shuffle=True) for cl in range(num_client)]
    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = args.warmUp and epoch < 5
        if atWarmup:
            sf.lr_warm_up(optimizers, epoch, args.lr)
            for run in range(runs):
                for cl in range(num_client):
                    local_iteration(trainloaders[cl], net_users[cl], optimizers[cl], criterions[cl], device)
                sf.initialize_zero(net_ps)
                [sf.push_model(net, net_ps, num_client) for net in net_users]

        else:
            for run in range(runs):
                for cl in range(num_client):
                    local_iteration(trainloaders[cl], net_users[cl], optimizers[cl], criterions[cl], device)

                ps_model_flat = sf.get_model_flattened(net_ps, device)
                ps_model_dif = torch.zeros_like(ps_model_flat)

                for cl in range(num_client):
                    difmodel = sf.get_model_flattened(net_users[cl],device).sub(ps_model_flat)
                    difmodel.add_(errors[cl])
                    worker_mask = torch.zeros_like(difmodel,device=device)
                    worker_mask[difmodel.abs().topk(k=k,dim=0)[1]] = 1
                    ps_model_dif.add_(difmodel.mul(worker_mask),alpha=1/num_client)
                    errors[cl] = difmodel.mul(1-worker_mask)

                ##server-side###
                ps_model_flat.add_(ps_model_dif)
                sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
                [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
                #################

        acc = evaluate_accuracy(net_ps, testloader, device)
        [schedulers[cl].step() for cl in range(num_client)]
        lr_ = sf.get_LR(optimizers[0])
        if lr_ != currentLR and not atWarmup:
            [error.mul_(lr_ / currentLR) for error in errors]
        currentLR = lr_
        accuracys.append(acc * 100)
        print('accuracy:', acc * 100)
    return accuracys