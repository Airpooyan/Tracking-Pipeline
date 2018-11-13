import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import warnings

from torch.autograd import Variable

from track.alexnet_old import SiameseAlexNet
from track.config import config
from track.utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image
from track.custom_transforms import Normalize, ToTensor

torch.set_num_threads(1) # otherwise pytorch will take all cpus

class SiamFCTracker:
	def __init__(self, gpu_id=0, model_path='/export/home/zby/SiamFC/models/output/siamfc_35_old.pth'):
		self.gpu_id = gpu_id
		print('----------------------------------------')
		print('Tracker: Using old tracker')
		print('Tracker: Initilizing tracking network...')
		with torch.cuda.device(gpu_id):
			self.model = SiameseAlexNet(gpu_id, train=False)
			print('Tracker: Loading checkpoint from %s'%(model_path))
			self.model.load_state_dict(torch.load(model_path))
			self.model = self.model.cuda()
			self.model.eval() 
			print('Tracker: Tracking network has been initilized')
		self.transforms = transforms.Compose([
				ToTensor()
				])
		self.data_dict = dict()

	def _cosine_window(self, size):
		"""
			get the cosine window
		"""
		cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
		cos_window = cos_window.astype(np.float32)
		cos_window /= np.sum(cos_window)
		return cos_window
		
	def see_data_dict(self):
		return self.data_dict
	
	def clear_data(self):
		self.data_dict = dict()
		
	def delete_id(self, track_id):
		if track_id in self.data_dict:
			del self.data_dict[track_id]
				
	def update_data_dict(self, frame, bbox, id):
		self.delete_id(id)
		im = frame
		id_dict = dict()
		id_dict['bbox'] = (bbox[0], bbox[1], bbox[2], bbox[3]) # zero based
		id_dict['pos'] = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])  # center x, center y, zero based
		id_dict['target_sz'] = np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])                            # width, height
		# get exemplar img
		id_dict['img_mean'] = tuple(map(int, frame.mean(axis=(0, 1))))
		exemplar_img, scale_z, s_z = get_exemplar_image(im, id_dict['bbox'],
				config.exemplar_size, config.context_amount, id_dict['img_mean'])
		exemplar_img = self.transforms(exemplar_img)[None,:,:,:]   # add new axis

		# get exemplar feature
		with torch.no_grad():
			with torch.cuda.device(self.gpu_id):
				exemplar_img_var = Variable(exemplar_img.cuda())
				feature = self.model((exemplar_img_var, None))
				id_dict['feature'] = feature

		self.penalty = np.ones((config.num_scale)) * config.scale_penalty
		self.penalty[config.num_scale//2] = 1

		# create cosine window
		self.interp_response_sz = config.response_up_stride * config.response_sz
		self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

		# create scalse
		self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
				np.floor(config.num_scale/2)+1)

		# create s_x
		id_dict['s_x'] = s_z + (config.instance_size-config.exemplar_size) / scale_z

		# arbitrary scale saturation
		id_dict['min_s_x'] = 0.2 * id_dict['s_x']
		id_dict['max_s_x'] = 5 * id_dict['s_x']
		self.data_dict[id] = id_dict
		#print(id_dict)
	
	def track_all(self, frame):
		bbox_list = []
		for id in self.data_dict:
			bbox, score = self.track_id(frame, id)
			bbox_list.append(bbox +[score] +[id])
		return bbox_list
			
	def track_id(self, frame, id):
		"""track object based on the previous frame
		Args:
			frame: an RGB image

		Returns:
			bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
		"""
		im = frame
		id_dict = self.data_dict[id]
		size_x_scales = id_dict['s_x'] * self.scales
		pyramid = get_pyramid_instance_image(im, id_dict['pos'], config.instance_size, size_x_scales, id_dict['img_mean'])
		instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0)
		with torch.no_grad():
			with torch.cuda.device(self.gpu_id):
				instance_imgs_var = Variable(instance_imgs.cuda())
				response_maps = self.model((None, instance_imgs_var), feature=id_dict['feature'])
				response_maps = response_maps.data.cpu().numpy().squeeze()
		response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
			 for x in response_maps]
		# get max score
		max_score = np.array([x.max() for x in response_maps_up]) * self.penalty
		# penalty scale change
		#score_max = max_score.max()
		scale_idx = max_score.argmax()
		response_map = response_maps_up[scale_idx]
		response_map -= response_map.min()
		response_map /= response_map.sum()
		response_map = (1 - config.window_influence) * response_map + \
				config.window_influence * self.cosine_window
		score_max = response_map.max()*10000
		max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
		#print(max_r, max_c)
		# displacement in interpolation response
		disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
		# displacement in input
		disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
		# displacement in frame
		scale = self.scales[scale_idx]
		disp_response_frame = disp_response_input * (id_dict['s_x'] * scale) / config.instance_size
		# position in frame coordinates
		id_dict['pos'] += disp_response_frame
		# scale damping and saturation
		id_dict['s_x'] *= ((1 - config.scale_lr) + config.scale_lr * scale)
		id_dict['s_x'] = max(id_dict['min_s_x'], min(id_dict['max_s_x'], id_dict['s_x']))
		id_dict['target_sz'] = ((1 - config.scale_lr) + config.scale_lr * scale) * id_dict['target_sz']
		bbox = [id_dict['pos'][0] - id_dict['target_sz'][0]/2 + 1, # xmin   convert to 1-based
				id_dict['pos'][1] - id_dict['target_sz'][1]/2 + 1, # ymin
				id_dict['pos'][0] + id_dict['target_sz'][0]/2 + 1, # xmax
				id_dict['pos'][1] + id_dict['target_sz'][1]/2 + 1] # ymax
		return bbox,score_max
