from torch import nn

from .transformer import CBAMLayer

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		self.fc = nn.Sequential(
			nn.Linear(64 * 7 * 7, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(-1, 64 * 7 * 7)
		y = self.fc(x)
		return y

class CnnWithCbam(CNN):
	def __init__(self):
		super(CnnWithCbam, self).__init__()
		self.cbam1 = CBAMLayer(32)
		self.cbam2 = CBAMLayer(64)

	def forward(self, x):
		x = self.conv1(x)
		x = self.cbam1(x)
		x = self.conv2(x)
		x = self.cbam2(x)
		x = x.view(-1, 64*7*7)
		y = self.fc(x)
		return y