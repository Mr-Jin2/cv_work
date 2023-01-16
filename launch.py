import os

import torch
from torch import nn
from torch import optim

from utils.common import Dict2Obj
from utils.log import setup_logger
from models.cnn import CnnWithCbam, CNN
from loguru import logger

from dataset import mk_loader
from chapter.attention.cbam_exp import expriment

def train(args):
	# dataset loader
	train_loader, test_loader = mk_loader(name=args.datasetname,
										  batch_size=args.batch_size)

	device = args.device
	optimizer = args.optimizer
	net = args.net.to(device=device)
	batch_size = args.batch_size
	ckpt_dir = args.ckpt_dir

	# train
	net.train()
	for epoch in range(args.epoch):
		running_loss = 0.0
		for i, data in enumerate(train_loader):
			inputs, labels = data[0].to(device), data[1].to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = args.loss_fn(outputs, labels)
			loss.backward()
			args.optimizer.step()
			running_loss += loss.item()

			# avoid oom
			del inputs, labels, outputs, loss
			torch.cuda.empty_cache()

			if i % 100 == 99:
				logger.info('[%d,%5d] loss :%.3f' %
							(epoch + 1, i + 1, running_loss / batch_size))
				running_loss = 0.0

		# save ckpt each epoch
		torch.save(net.state_dict(), f'{ckpt_dir}/{net.__class__.__name__}-e-{epoch + 1}.ckpt')

		# valid
		correct = 0
		total = 0
		net.eval()
		with torch.no_grad():
			for data in test_loader:
				inputs, labels = data[0].to(device), data[1].to(device)
				outputs = net(inputs)
				_, predicts = torch.max(outputs.data, 1)
				total += labels.size(0)  # labels 的长度
				correct += (predicts == labels).sum().item()  # 预测正确的数目

		logger.info('epoch %d Test accuracy images: %d %%' % (epoch + 1,
															  100 * correct / total))

	print('Finished Training')
	return net


def initial():
	setup_logger(save_dir='log/')
	os.makedirs('ckpt', exist_ok=True)
	os.makedirs('log', exist_ok=True)


if __name__ == '__main__':

	ckpt_dir = 'ckpt'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	BATCH_SIZE = 64
	EPOCH = 10
	lr = 0.01
	momentum = 0.9
	datasetname = 'fashionminist'

	# net = CnnWithCbam()
	net = CNN()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
	args = dict(
		device=device,
		batch_size=BATCH_SIZE,
		epoch=EPOCH,
		lr=lr,
		momentum=momentum,
		datasetname=datasetname,
		net=net,
		loss_fn=loss_fn,
		optimizer=optimizer,
		ckpt_dir=ckpt_dir
	)
	args=Dict2Obj(args)

	# train(args)

	model = CNN()
	expriment(model, datasetname, 'ckpt/CNN-e-10.ckpt')