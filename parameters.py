import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='gpu_id')
    parser.add_argument('--num_trail', type=int, default=3, help='number of trail')
    parser.add_argument('--mode', type=str, default='tcs', help='tcs or topk')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='dirichlet', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--alfa', type=int, default=50, help='alfa for dirichlet')

    # Federated params
    parser.add_argument('--num_client', type=int, default=10, help='number of clients')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.5, help='learning_rate')
    parser.add_argument('--layer_wise_spars', type=bool, default=False, help='include layer-wise sparsification')
    parser.add_argument('--worker_LWS', type=bool, default=False, help='include layer-wise sparsification')
    parser.add_argument('--lws_sparsity', type=int, default=1000, help= 'divide individual layers with this value')
    parser.add_argument('--lws_sparsity_w', type=int, default=2500, help='divide individual layers in worker with this value')
    parser.add_argument('--sparsity_window', type=int, default=100, help='largest grad entry is chosen within this window')
    parser.add_argument('--lr_change', type=list, default=[150, 225], help='determines the at which epochs lr will decrease')
    parser.add_argument('--warmUp', type=bool, default=True, help='LR warm up.')
    parser.add_argument('--biasFairness', type=bool, default=False, help='LR warm up.')
    parser.add_argument('--LSGDturn', type=int, default=4,help='largest grad entry is chosen within this window')

    # Quantization params
    parser.add_argument('--quantization', type=bool, default=False, help='apply quantization or not')
    parser.add_argument('--num_groups', type=int, default=16, help='Number Of groups')
    args = parser.parse_args()
    return args