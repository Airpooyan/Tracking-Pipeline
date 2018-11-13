import torch
import cv2
import _init_paths
from PIL import Image
from torchvision import transforms
from embedding.model.Model import Model
from embedding.utils.utils import load_state_dict
import time
#from utils import load_state_dict


class EmbeddingNet():
	def __init__(self, gpu_id, model_path):
		self.gpu_id = gpu_id
		self.model_path = model_path
		# self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
		with torch.cuda.device(gpu_id):
			self.model = Model(last_conv_stride=1)
			print('---------------------------------')
			print('Load the embedding model from {}'.format(model_path))
			weight = torch.load(model_path)
			load_state_dict(self.model, weight)
			self.model=self.model.cuda()
			self.model.eval()
			print('Embedding model has been initialized')
		self.transform = transforms.Compose([
			transforms.ToTensor()
		])

	def embedding(self, frame, bbox):
		#print('==' * 10)
		#print('Get the feature of the bbox')
		start = time.time()

		image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		x_min = bbox[0]
		y_min = bbox[1]
		x_max = bbox[2]
		y_max = bbox[3]
		person_im = image.crop((x_min, y_min, x_max, y_max))
		person_im_t = transforms.ToTensor()(person_im)
		person_im_t = person_im_t.unsqueeze_(0)

		second = time.time()
		#print('Process takes {}s'.format(second-start))
		with torch.no_grad():
			person_im_t = person_im_t.cuda()
			feature = self.model(person_im_t)
		feature = feature.to('cpu').detach().numpy()[0]
		feature_list = feature.tolist()
		#print('Forward takes {}s'.format(time.time()-second))
		return feature_list


if __name__ == '__main__':
    gpu_id = 0
    model_path = '/export/home/zby/SiamFC/models/embedding_model.pth'
    image_path = '/export/home/zby/SiamFC/data/posetrack/images/train/000001_bonn_train/000075.jpg'
    img = cv2.imread(image_path)
    bbox = [1,2,3,4]
    en = EmbeddingNet(gpu_id, model_path)
    start = time.time()
    feature = en.get_feature(img, bbox)
    print('Embedding has taken {}s'.format(time.time()-start))
    print('The feature of the image is {}'.format(len(feature)))
