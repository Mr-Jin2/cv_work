import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

dataset_map = dict(
	minist=torchvision.datasets.MNIST,
	fashionminist=torchvision.datasets.FashionMNIST
)


def mk_loader(name, batch_size, shuffle=True):
	assert dataset_map.get(name), f'dataset {name} not exist'
	dataset = dataset_map[name]
	train = dataset(root='dataset/data',
					train=True,
					download=True,
					transform=transforms.ToTensor()
					)
	test = dataset(root='dataset/data',
					train=False,
					download=True,
					transform=transforms.ToTensor()
					)
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
	test_laoder = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

	return train_loader, test_laoder