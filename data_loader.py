""""""
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import math


def get_mnist_dataset():
    """returns trainset and testsets for MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_fmnist_dataset():
    """returns trainset and testsets for Fashion MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_cifar10_dataset():
    """returns trainset and testsets for Fashion CIFAR10 dataset"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_dataset(args):

    dataset_name = args.dataset_name

    if dataset_name == 'mnist':
        trainset, testset = get_mnist_dataset()
    elif dataset_name == 'fmnist':
        trainset, testset = get_fmnist_dataset()
    elif dataset_name == 'cifar10':
        trainset, testset = get_cifar10_dataset()
    else:
        raise ValueError('dataset name can only be mnist, fmnist or cifar10')

    return trainset, testset


def get_indices(trainset, args):
    """returns the indices of sample for each worker in either iid, or non_iid manner provided in args"""

    if args.dataset_dist == 'iid':
        inds = get_iid_index(trainset, args)
    elif args.dataset_dist == 'non_iid':
        inds = get_non_iid_index(trainset, args)
    elif args.dataset_dist == 'dirichlet':
        inds = dirichlet_dist(trainset, args)
    else:
        raise ValueError('Dataset distribution can only be iid or non_iid')
    return inds


def get_non_iid_index(trainset, args):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a non_iid distribution. Sorts the indexs that have a lablel 0 to label 10. Then equally splits
     the indexes for each user"""
    if args.dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif args.dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)

    inds_sorted = np.argsort(labels)
    num_sample_perworker = int(num_samples / args.num_client)

    indx_sample = {n: [] for n in range(args.num_client)}

    for user in range(args.num_client):
        indx_sample[user] = inds_sorted[user * num_sample_perworker: (user + 1) * num_sample_perworker]

    return indx_sample

def dirichlet_dist(trainset, args):
    labels = None
    try:
        labels = trainset.train_labels
    except:
        labels = trainset.targets
    finally:
        if labels is None:
            raise NotImplementedError('No labels')

    num_cls = max(labels) + 1
    inds_sorted = np.argsort(labels)
    train_sorted = []
    for i in range(num_cls):
        total_sample = np.sum(np.asarray(labels) == i)
        train_sorted.append(inds_sorted[i * total_sample:(i + 1) * total_sample])

    dirichlet = np.repeat(args.alfa, num_cls)
    dirichlet_vec = np.random.dirichlet(dirichlet, args.num_client)
    clas_dist = np.sum(dirichlet_vec, axis=0)
    dist = dirichlet_vec
    for worker in dist:
        for cls in range(num_cls):
            worker[cls] *= 1/clas_dist[cls]

    print(np.around(dist,2))

    indx_sample = {n: [] for n in range(10)}

    for cls in range(num_cls):
        norm_val = 1 / clas_dist[cls]
        start_ind = 0
        cls_samples = len(train_sorted[cls])
        for i, worker in enumerate(dirichlet_vec):
            if i == num_cls - 1:
                indx_sample[i] = np.concatenate((indx_sample[i], train_sorted[cls][start_ind:]), axis=0)
            else:
                cls_alf = worker[cls]
                end_ind = start_ind + math.floor(cls_samples * cls_alf * norm_val)
                indx_sample[i] = np.concatenate((indx_sample[i], train_sorted[cls][start_ind:end_ind]), axis=0)
                start_ind = end_ind
    for sample in indx_sample:
        indx_sample[sample] =np.asarray(indx_sample[sample],dtype='int')
    return indx_sample

def get_iid_index(trainset, args):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a iid distribution. Then equally splits
     the indexes for each user"""

    if args.dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif args.dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)
    num_user = args.num_client
    num_sample_perworker = int(num_samples / num_user)
    inds = [*range(num_samples)]
    inds_split = np.random.choice(inds, [num_user, num_sample_perworker], replace=False)
    indx_sample = {n: [] for n in range(num_user)}
    for user in range(num_user):
        indx_sample[user] = list(inds_split[user])

    return indx_sample


class DatasetSplit(Dataset):
    def __init__(self, dataset, indxs):
        self.dataset = dataset
        self.indxs = indxs

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.indxs[item]]
        return image, label

