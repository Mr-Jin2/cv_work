import os

import cv2
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

dataset_map = dict(
	minist=torchvision.datasets.MNIST,
	fashionminist=torchvision.datasets.FashionMNIST
)


class ClsDataset(Dataset):
	def __init__(self, imgdir, labelfile, transform, train=True, train_ratio=0.8, suffix='.jpg'):
		super(ClsDataset, self).__init__()
		self.imgdir = imgdir
		self.labelfile = labelfile
		self.transform = transform
		self.train = train,
		self.labels = pd.read_csv(self.labelfile)
		self.rows = self.labels.shape[0]
		self.split = int(self.rows * train_ratio)
		self.suffix = suffix

	def __getitem__(self, item):
		if not self.train:
			item = item + self.split
		imgname = str(self.labels['id'][item]) + self.suffix
		target = int(self.labels['target'][item])
		data = cv2.imread(os.path.join(self.imgdir, imgname))
		if self.transform:
			data = self.transform(data)
		return data, target

	def __len__(self):
		if self.train:
			return self.split
		else:
			return self.rows - self.split


def custom_dataset_loader(
		batch_size,
		imgdir,
		labelfile,
		transform,
		shuffle=True
):

	train = ClsDataset(imgdir, labelfile, transform, train=True)
	test = ClsDataset(imgdir, labelfile, transform, train=False)
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
	test_laoder = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
	return train_loader, test_laoder


def origin_dataset_loader(name, batch_size, shuffle=True):
	assert dataset_map.get(name), f'dataset {name} not exist'
	dataset = dataset_map[name]
	train = dataset(root='~/PycharmProject/cv_work/dataset/data',
					train=True,
					download=True,
					transform=transforms.ToTensor()
					)
	test = dataset(root='~/PycharmProject/cv_work/dataset/data',
				   train=False,
				   download=True,
				   transform=transforms.ToTensor()
				   )
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
	test_laoder = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

	return train_loader, test_laoder


if __name__ == '__main__':
	df = pd.read_csv('/home/jyz/Downloads/boolart-image-classification/train.csv')
	x = df['target']
	print(set(x))
