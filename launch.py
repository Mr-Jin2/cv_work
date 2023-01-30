import os

import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from loguru import logger

from utils.common import Dict2Obj
from utils.log import setup_logger
from models.cnn import CnnWithCbam, CNN
from models.resnet import resnet50
from dataset import origin_dataset_loader, custom_dataset_loader
from chapter.attention.exp import expriment
from chapter.cnn_backbone.exp import expriment


def train(args):
	device = args.device
	optimizer = args.optimizer
	net = args.net.to(device=device)
	batch_size = args.batch_size
	ckpt_dir = args.ckpt_dir

	# train
	net.train()
	for epoch in range(args.epoch):
		running_loss = 0.0
		for i, (data, targets) in enumerate(args.train_loader):
			inputs = data.to(args.device)
			targets = targets.to(args.device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = args.loss_fn(outputs, targets)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			# avoid oom
			del inputs, targets, outputs, loss
			torch.cuda.empty_cache()

			if i % 100 == 99:
				logger.info('[%d,%5d] loss :%.3f' %
							(epoch + 1, i + 1, running_loss / batch_size))
				running_loss = 0.0

		# 调整学习率
		args.scheduler_lr.step()

		# save ckpt each epoch
		torch.save(net.state_dict(), f'{ckpt_dir}/{net.__class__.__name__}-e-{epoch + 1}.ckpt')

		# valid
		correct = 0
		total = 0
		net.eval()
		with torch.no_grad():
			for data, targets in args.test_loader:
				inputs, targets = data.to(device), targets.to(device)
				outputs = net(inputs)
				_, predicts = torch.max(outputs.data, 1)
				total += targets.size(0)  # labels 的长度
				correct += (predicts == targets).sum().item()  # 预测正确的数目

		logger.info('epoch %d Test accuracy images: %d %%' % (epoch + 1,
															  100 * correct / total))

	print('Finished Training')
	return net


def initial():
	setup_logger(save_dir='log/')
	os.makedirs('ckpt', exist_ok=True)
	os.makedirs('log', exist_ok=True)


if __name__ == '__main__':
	initial()
	ckpt_dir = 'ckpt'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	BATCH_SIZE = 64
	EPOCH = 10
	lr = 0.01
	momentum = 0.9

	# # data loader
	# datasetname = 'fashionminist'
	# train_loader, test_loader \
	# 	= origin_dataset_loader(name=datasetname,
	# 							batch_size=BATCH_SIZE)

	imgdir = '/home/jyz/Downloads/boolart-image-classification/train_image/'
	labelfile = '/home/jyz/Downloads/boolart-image-classification/train.csv'
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Resize(size=[64, 64])]
	)

	train_loader, test_loader = custom_dataset_loader(BATCH_SIZE, imgdir, labelfile, transform)

	# # model
	# net = CnnWithCbam()
	# net = CNN()
	# net = torchvision.models.resnet50(num_classes=44)
	# net = resnet50()
	net = torchvision.models.efficientnet_b5()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
	scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	args = dict(
		device=device,
		batch_size=BATCH_SIZE,
		epoch=EPOCH,
		lr=lr,
		momentum=momentum,
		train_loader=train_loader,
		test_loader=test_loader,
		net=net,
		loss_fn=loss_fn,
		optimizer=optimizer,
		scheduler_lr=scheduler_lr,
		ckpt_dir=ckpt_dir
	)
	args = Dict2Obj(args)

	train(args)

	# ckpt = 'ckpt/ResNet-e-10.ckpt'
	ckpt = 'ckpt/EfficientNet-e-10.ckpt'
	expriment(net, ckpt)
