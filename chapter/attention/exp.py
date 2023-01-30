import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.grad_cam import GradCAM, show_cam_on_image
from dataset.data_loader import origin_dataset_loader
from models.cnn import CnnWithCbam

def sample_one_data(dataloader, label_id):
	for data, label in dataloader:
		if label == label_id:
			label = label.item()
			data = data.numpy()
			data = np.squeeze(data, axis=0)
			print(label)
			return data, label

def gen_heat_map(img, model):

	target_layers = [model.cbam1, model.cbam2]

	# data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	data_transform = lambda x:torch.Tensor(x)

	# [C, H, W]
	img_tensor = data_transform(img)
	# expand batch dimension
	# [C, H, W] -> [N, C, H, W]
	input_tensor = torch.unsqueeze(img_tensor, dim=0)

	cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
	target_category = 2

	grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

	grayscale_cam = grayscale_cam[0, :]
	visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
									  grayscale_cam,
									  use_rgb=True)
	return visualization

def expriment(model, datasetname, pretrained_model):
	BATCH_SIZE = 1

	_, test_loader = mk_loader(datasetname, BATCH_SIZE, shuffle=False)
	model.load_state_dict(torch.load(pretrained_model))

	for i in range(10):
		data, label = sample_one_data(test_loader, i)  # c,h,w
		heatmap = gen_heat_map(data, model)
		data = data.transpose((1, 2, 0))
		plt.subplot(10, 2, i * 2 + 1)
		plt.imshow(data)

		plt.subplot(10, 2, i * 2 + 2)
		plt.imshow(heatmap)
		plt.xticks([])
		plt.yticks([])
	plt.show()


