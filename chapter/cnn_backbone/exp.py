import os
import torch
import torchvision
from torchvision import transforms
import pandas as pd
import cv2


def expriment(model, ckpt):
	testimg_dir = '/home/jyz/Downloads/boolart-image-classification/test_image/'
	test_item = '/home/jyz/Downloads/boolart-image-classification/sample_submission.csv'
	report = test_item.replace('sample_submission', 'sample_submission_bak')
	model.load_state_dict(torch.load(ckpt))
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(size=[64, 64])
	])

	test_df = pd.read_csv(test_item).copy()
	model.eval()
	for i, imgname in enumerate(test_df['id']):
		imgname = str(imgname) + '.jpg'
		img_path = os.path.join(testimg_dir, imgname)
		img = cv2.imread(img_path)
		data = transform(img)
		data = torch.unsqueeze(data, dim=0)
		pred = model(data)
		pred_id = torch.argmax(pred, 1).item()
		test_df['predict'][i] = str(pred_id)
	test_df.to_csv(report, index=False)


if __name__ == '__main__':
	pass