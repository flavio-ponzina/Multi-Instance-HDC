import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import *
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchhd.datasets import ISOLET, UCIHAR, Cardiotocography10Clases
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as T

def load_dataset(dataset="mnist", bsize=512):
	if(dataset == "mnist"):
		transform = T.Compose([T.ToTensor(), T.Normalize((0.1307), (0.3081))])
		train_ds = MNIST("data/", train=True, download=True, transform=transform)
		test_ds = MNIST("data/", train=False, download=True, transform=transform)
		fdim = 784
		num_classes = len(train_ds.classes)
	elif(dataset =="fmnist"):
		transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.2860], std=[0.3530])])
		train_ds = FashionMNIST("data/", train=True, download=True, transform=transform)
		test_ds = FashionMNIST("data/", train=False, download=True, transform=transform)
		fdim = 784
		num_classes = len(train_ds.classes)
	elif dataset == 'cifar10':
		transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
		# transform = T.ToTensor()
		train_ds = CIFAR10("../data", train=True, transform=transform, download=True)
		test_ds  = CIFAR10("../data", train=False, transform=transform, download=True)
		fdim = 3072
		num_classes = len(train_ds.classes)
	elif dataset == 'isolet':
		# transform = Normalize()
		train_ds = ISOLET("../data", train=True, download=True)
		test_ds  = ISOLET("../data", train=False, download=True)
		fdim = 617
		num_classes = len(train_ds.classes)
	elif dataset == 'ucihar':
		# transform = Normalize()
		train_ds = UCIHAR("../data", train=True, download=True)
		test_ds  = UCIHAR("../data", train=False, download=True)
		fdim = 561
		num_classes = len(train_ds.classes)
	elif dataset == 'cardio':
		train_ds_list = []
		test_ds_list = []
		for fold_id in range(3):
			train_ds_list.append(Cardiotocography10Clases("../data", train=True, download=True,fold=fold_id))
		test_ds_list.append(Cardiotocography10Clases("../data", train=False, download=True,fold=3))
		train_ds = ConcatDataset(train_ds_list)
		test_ds  = ConcatDataset(test_ds_list)
		fdim = 21
		num_classes = len(train_ds_list[0].classes)

	

	train_dl = DataLoader(train_ds, bsize, shuffle=True)
	test_dl = DataLoader(test_ds, bsize, shuffle=False)
	_tr_len = len(train_dl)
	_ts_len = len(test_dl)


	return train_ds, train_dl, test_dl, _tr_len, _ts_len, fdim, num_classes



def bagging_train_load_dataset(train_ds, bsize=512):
	train_sampler = RandomSampler(train_ds, replacement=True, num_samples=int(1*len(train_ds)))
	train_dl = DataLoader(train_ds, bsize, sampler=train_sampler)
	return train_dl


def boosting_train_load_dataset(train_ds, sample_weights, bsize=512):
	train_sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
	train_dl = DataLoader(train_ds, bsize, sampler=train_sampler)
	return train_dl
