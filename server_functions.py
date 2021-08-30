import torch
import math
import time
import numpy as np
import torch.nn as nn

def pull_model(model_user, model_server):

    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_user.data = param_server.data[:] + 0

    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zero_grad_ps(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)

    return None


def push_grad(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.grad.data += param_user.grad.data / num_cl
    return None

def push_model(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.data += param_user.data / num_cl
    return None

def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def update_model(model, lr):
    for param in model.parameters():
        param.data.add_(-lr, param.grad.data)
    return None


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened

def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened

def get_model_sizes(model):
    # get the size of the layers and number of eleents in each layer.
    # only layers that are trainable
    net_sizes = []
    net_nelements = []
    for p in model.parameters():
        if p.requires_grad:
            net_sizes.append(p.data.size())
            net_nelements.append(p.nelement())
    return net_sizes, net_nelements




def get_indices(net_sizes, net_nelements):
    # for reconstructing grad from flattened grad
    ind_pairs = []
    ind_start = 0
    ind_end = 0
    for i in range(len(net_sizes)):

        for j in range(i + 1):
            ind_end += net_nelements[j]
        # print(ind_start, ind_end)
        ind_pairs.append((ind_start, ind_end))
        ind_start = ind_end + 0
        ind_end = 0
    return ind_pairs



def make_model_unflattened(model, model_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        temp = model_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
        p.data = temp.reshape(net_sizes[i])
        i += 1
    return None


def adjust_learning_rate(optimizer, epoch,lr_change, lr):

    lr_change = np.asarray(lr_change)
    loc = np.where(lr_change == epoch)[0][0] +1
    lr *= (0.1**loc)
    lr = round(lr,3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_warm_up(optimizers, epoch,start_lr):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            if epoch == 0:
                param_group['lr'] = 0.1
            else:
                lr_change = (start_lr - 0.1) / 4
                param_group['lr'] = (lr_change * epoch) + 0.1
def get_LR(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr





def sparse_timeC(flat_params, exclusive_sparsity_windows, prev_ps_mask, device):
    exclusive_sparse= math.ceil(flat_params.numel() / (exclusive_sparsity_windows))
    worker_exclusives = flat_params.mul(1-prev_ps_mask).to(device)
    excl_tops, excl_ind = torch.topk(worker_exclusives.abs(), k=exclusive_sparse, dim=0)
    exclusive_mask = torch.zeros_like(flat_params,device=device)
    exclusive_mask[excl_ind] = 1
    mask=prev_ps_mask.add(exclusive_mask)
    flat_params *= mask
    return None

def sparse_pool(flat_params, sparsity_window,exclusive_sparsity_windows, prev_ps_mask, device):
    exclusive_sparse= math.ceil(flat_params.numel() / (exclusive_sparsity_windows))
    primary_sparsity = math.ceil(flat_params.numel() / (sparsity_window))
    mask = torch.zeros_like(flat_params,device=device)
    ps_vals,ps_inds = flat_params.mul(prev_ps_mask).abs().topk(k=primary_sparsity,dim=0)
    mask[ps_inds] = 1
    w_vals, w_inds = flat_params.mul(1-mask).abs().topk(k=exclusive_sparse,dim=0)
    mask[w_inds] = 1
    flat_params.mul_(mask)
    return None


def sparse_timeC_alt(grad_flat,exclusive_sparsity_windows,
    layer_spar,prev_ps_mask,ind_pairs,biasFairness,layersNames,device):
    exclusive_sparse= math.ceil(grad_flat.numel()/exclusive_sparsity_windows)
    exclusive_mask = 1 - prev_ps_mask
    exclusive_grads = (grad_flat.mul(exclusive_mask)).to(device)
    inds = torch.empty(0,dtype=torch.float).to(device)
    worker_mask = torch.zeros_like(grad_flat)
    for layer,layer_n in zip(ind_pairs,layersNames):
        if not biasFairness or 'bias' not in layer_n:
            startPoint= (layer[0])
            endPoint= (layer[1])
            layer_len = endPoint - startPoint
            l_top_k = math.ceil(layer_len / layer_spar)
            l_vals, l_ind = torch.topk((exclusive_grads[startPoint:endPoint]).abs(), k=l_top_k, dim=0)
            l_ind.add_(startPoint)
            inds = torch.cat((inds.float(), l_ind.float()), 0)
    inds = inds.long()
    if exclusive_sparse > inds.numel():
        clone_worker_grad = torch.clone(exclusive_grads)
        clone_worker_grad[inds] = 0
        topk = exclusive_sparse - inds.numel()
        inds_ = torch.topk(clone_worker_grad.abs(),k=topk,dim=0)[1]
        inds = torch.cat((inds, inds_), 0)
    worker_mask[inds] = 1
    worker_mask += prev_ps_mask
    grad_flat *= worker_mask
    return None

def sparse_special_mask(flat_grad,sparsity_window,layer_spar,ind_pairs,biasFairness,layerNames,device):
    inds = torch.empty(0).to(device)
    for layer, layer_n in zip(ind_pairs, layerNames):
        if not biasFairness or 'bias' not in layer_n:
            startPoint = (layer[0])
            endPoint = (layer[1])
            layer_len = endPoint - startPoint
            l_top_k = math.ceil(layer_len / layer_spar)
            l_vals, l_ind = torch.topk((flat_grad[startPoint:endPoint]).abs(), k=l_top_k, dim=0)
            l_ind.add_(startPoint)
            inds = torch.cat((inds.float(), l_ind.float()), 0)
    inds = inds.long()
    clone_grad = torch.clone(flat_grad).to(device)
    clone_grad[inds] = 0
    topk = math.ceil(len(flat_grad)/(sparsity_window)) - inds.numel()
    vals_,inds_ = torch.topk(clone_grad.abs(),k=topk,dim=0)
    inds = torch.cat((inds, inds_), 0)
    clone_grad *=0
    clone_grad[inds] = 1
    return clone_grad


def quantize(params_flat, args, device):
    sparseCount = torch.sum(params_flat != 0).int()
    vals, ind = torch.topk(params_flat.abs(), k=sparseCount, dim=0)
    custom_denom = (vals[0] / vals[sparseCount-1]) ** (1/args.num_groups)
    top_Val = vals[0]
    sign_mask = torch.sign(params_flat)
    clone_grad = torch.zeros_like(params_flat).to(device)
    clone_Vals = torch.clone(vals).to(device)
    for i in range(args.num_groups):
        mask = (clone_Vals <= top_Val) * (clone_Vals > (top_Val / custom_denom))
        clone_grad[torch.masked_select(ind,mask)] = torch.mean(torch.masked_select(vals,mask))
        top_Val /= custom_denom
    clone_grad.mul_(sign_mask)
    return clone_grad

def get_BN_mask(net,device):
    mask = torch.empty(0).to(device)
    for layer in net.modules():  # Prune only convolutional and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer_weight = layer.weight
            len = layer_weight.numel()
            mask_ = torch.zeros(len,device=device)
            mask = torch.cat((mask, mask_), 0)
            if layer.bias is not None:
                bias = layer.bias.numel()
                mask_ = torch.ones(bias, device=device)
                mask = torch.cat((mask, mask_), 0)
        elif isinstance(layer, nn.BatchNorm2d):
            bn_params = 0
            for p in layer.parameters():
                bn_params += p.numel()
            mask_ = torch.ones(bn_params, device=device)
            mask = torch.cat((mask, mask_), 0)
    mask = mask.int()
    return mask