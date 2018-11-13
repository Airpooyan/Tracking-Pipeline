import os
import json
import torch
import time
import numpy as np
import cv2
import scipy.spatial.distance as distance
# from log import Log
from torchvision import transforms
import _init_paths
from pose.core.config import config as cfg
from pose.core.config import update_config
from pose.core.config import update_dir
from pose.core.config import get_model_name
from pose.models.pose_resnet import get_pose_net

#from pose.core.inference import get_max_preds
#from pose.utils.transforms import flip_back
from pose.utils.transforms import get_affine_transform
#from pose.utils.transforms import transform_preds
#from pose.utils.utils import create_logger
from tqdm import tqdm
import random
import argparse

def parseArgs():
	parser = argparse.ArgumentParser(description="Evaluation of Embedding (PoseTrack)")
	parser.add_argument("-d", "--distance_thresh",dest = 'dis_thresh',required=False, default=0.1, type= float)
	return parser.parse_args()
	

class EmbeddingEvaluate:			
	def __init__(self, data_path, annotation_path, num_video,
					   gpu_id=0, model_path='/export/home/zby/SiamFC/data/models/final_new.pth.tar'):
		self.cfg_file='/export/home/zby/SiamFC/cfgs/pose_res152.yaml'
		update_config(self.cfg_file)
		self.data_path = data_path
		self.annotation_path = annotation_path
		self.num_video = num_video
		print('----------------------------------------')
		print('Pose & Embedding: Initilizing pose & embedding network...')
		#self.test_thresh_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18 ,0.19, 0.2]
		self.test_thresh_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18 ,0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29 ,0.3, 1]
		self.gpu_id = gpu_id
		self.pixel_std = 200
		self.image_size = cfg.MODEL.IMAGE_SIZE
		self.image_width = self.image_size[0]
		self.image_height = self.image_size[1]
		self.aspect_ratio = self.image_width * 1.0 /self.image_height
		self.transform = transforms.Compose([transforms.ToTensor(),
											 transforms.Normalize(mean = [0.485,0.456,0.406],
																  std = [0.229,0.224,0.225])
											])		
		cfg.TEST.FLIP_TEST = True
		cfg.TEST.POST_PROCESS = True
		cfg.TEST.SHIFT_HEATMAP = True
		cfg.TEST.MODEL_FILE = model_path

		#cudnn.benchmark = cfg.CUDNN.BENCHMARK
		torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
		torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

		with torch.cuda.device(self.gpu_id):
			self.model = get_pose_net(cfg, is_train=False)
			self._load_model()
			self.model.eval()
		print('Pose & Embedding: Test thresh list is set as {}'.format(self.test_thresh_list))
	
	def _load_model(self,model_file = None):		
		model_file = cfg.TEST.MODEL_FILE if model_file is None else model_file
		print("Pose & Embedding: Loading checkpoint from %s" % (model_file))
		checkpoint=torch.load(model_file)
		from collections import OrderedDict
		model_dict = self.model.state_dict()
		new_state_dict = OrderedDict()
		for k,v in checkpoint.items():
			new_name = k[7:] if 'module' in k else k
			new_state_dict[new_name]=v
		model_dict.update(new_state_dict)
		self.model.load_state_dict(model_dict)
		self.model = self.model.cuda()
		print('Pose & Embedding: Pose & Embedding network has been initilized')	
	
	def extract_feature(self, im, bbox):
		im_in = im
		cords = bbox[0:4]
		center, scale = self.x1y1x2y2_to_cs(cords)
		r=0
		
		trans = get_affine_transform(center[0], scale[0], r, self.image_size)
		input_image = cv2.warpAffine(im_in, trans, (int(self.image_width), int(self.image_height)), flags=cv2.INTER_LINEAR)
		with torch.no_grad():
			input = self.transform(input_image)
			input = input.unsqueeze(0)
			with torch.cuda.device(self.gpu_id):
				input = input.cuda()
				feature = self.model(input, flag=1)
				if cfg.TEST.FLIP_TEST:
					input_flipped = np.flip(input.cpu().numpy(),3).copy()
					input_flipped = torch.from_numpy(input_flipped).cuda()
					feature_flipped = self.model(input_flipped, flag=1)
					feature = (feature + feature_flipped) *0.5
		
		feature = feature.cpu().numpy().squeeze()
		return feature
		
	def x1y1x2y2_to_cs(self, bbox):
		x,y,xmax,ymax = bbox
		w,h = xmax-x, ymax-y
		center = np.zeros((2), dtype=np.float32)
		center[0] = x + w * 0.5
		center[1] = y + h * 0.5

		if w > self.aspect_ratio * h:
			h = w * 1.0 / self.aspect_ratio
		elif w < self.aspect_ratio * h:
			w = h * self.aspect_ratio
		scale = np.array(
			[w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
			dtype=np.float32)
		if center[0] != -1:
			scale = scale * 1.25
		
		return np.expand_dims(center,0), np.expand_dims(scale,0)
	
	def xywh_to_x1y2x2y2(self, bbox):
		x, y, w, h = bbox[0:4]
		x1, y1, x2, y2 = x, y, x+w, y+h
		return [x1, y1, x2, y2]

	def _solve(self, annotation_single_path):
		#print('The annotation path is {}'.format(annotation_single_path))
		labeled_frames = []
		frames_bbox = {}
		with open(annotation_single_path, 'r') as f:
			json_obj = json.load(f)
			frames = json_obj.get('images')
			annotations = json_obj.get('annotations')

		for item_f in frames:
			if item_f.get('is_labeled'):
				labeled_frames.append(item_f)

		for item_a in annotations:
			file_name = item_a.get('file_name')
			if file_name in frames_bbox.keys():
				frames_bbox.get(file_name).append(item_a)
			else:
				frames_bbox[file_name] = []
				frames_bbox[file_name].append(item_a)

		return labeled_frames, frames_bbox

	def _create_video_list(self):
        # Base on the annotation's folder because the some training videos don't have the annotation
		random_video_list = []
		video_names = os.listdir(self.annotation_path)
		total_video = len(video_names)
		random_video_indexs = np.random.randint(total_video, size=self.num_video)
		for index in random_video_indexs:
			random_video_list.append(video_names[index][:-7])
		#print('The create random video list is {}'.format(random_video_list))
		return random_video_list
	
	def cos_distance(self, query, target):
		inner = np.dot(query, target)
		outer = np.linalg.norm(query)*np.linalg.norm(target)
		distance = (1 - inner / outer) * 0.5
		return distance
	
	def _single_search_l2(self, query, query_id, target, ids):
		dist = []
		#print(query.shape)
		for f in target:
			dist.append(self.cos_distance(query, f))
		#print(dist)
		dist = np.array(dist)
		index = dist.argsort(axis=0)[0]
		distance_min =  dist[index]
		top1 = ids[index]
		flag_array = np.zeros(len(self.test_thresh_list), dtype=np.int)
		for i,now_thresh in enumerate(self.test_thresh_list):
			flag_array[i] = False
			if distance_min > now_thresh:
				if not query_id in ids:
					flag_array[i] = True
			else:
				if top1 == query_id:
					flag_array[i] = True
		#print(flag_array)
		return flag_array

	def eval(self):
		video_list = self._create_video_list()
		right_cnt_array = np.zeros(len(self.test_thresh_list),dtype=np.int)
		pbar = tqdm(range(len(video_list)))
		all_flag = 0
		for video in video_list:
			pbar.update(1)
			#print('The video name is {}'.format(video))
			# The following operations are in the same video, so the **query_person_id** is just a number
			annotation_name = video + '_c.json'
			labeled_frames, frames_bbox = self._solve(self.annotation_path + annotation_name)

			# Can't from 0, we set the smallest frame, 10
			random_query_frame_index = np.random.randint(10, len(labeled_frames))
			random_query_frame = labeled_frames[random_query_frame_index]
			#print('The random_query_frame is {}'.format(random_query_frame['file_name']))
			bboxs = frames_bbox.get(random_query_frame.get('file_name'))
			random_people_index = np.random.randint(len(bboxs))
			query_person = bboxs[random_people_index]
			query_person_filename = query_person['file_name']
			quert_person_video, query_person_frame, query_person_id = query_person_filename.split('/')[-2], query_person_filename.split('/')[-1], query_person['track_id']
			#print('The query person video:{} frame:{} track_id:{}'.format(quert_person_video, query_person_frame, query_person_id))
			#pbar.set_description('Query person  video:{} frame:{} track_id:{:>2}'.format(quert_person_video, query_person_frame, query_person_id))

			random_target_frame_index = np.random.randint(0, random_query_frame_index)
			random_target_frame = labeled_frames[random_target_frame_index]
			target_people = frames_bbox.get(random_target_frame.get('file_name'))
			#print(target_people)
			
			im_path = self.data_path + video + '/' + random_query_frame.get('file_name').split('/')[-1]
			#print(im_path)
			query_image = cv2.imread(im_path)
			#query_person = self._crop(query_image, query_person.get('bbox'))  # The cropped query image

			if not 'bbox' in query_person:
				#print('The b-box is wrong')
				#pbar.set_description('The b-box is wrong')
				self.num_video -= 1
				continue
			query_bbox = self.xywh_to_x1y2x2y2(query_person['bbox'])
			query_feature = self.extract_feature(query_image, query_bbox)

			cache_images = []  # store the cropped image target images
			cache_ids = []    # store the cropped the images' id
			cache_features = []
			temp_image = cv2.imread(self.data_path + video + '/' + random_target_frame.get('file_name').split('/')[-1])
			for people in target_people:
				temp_bbox = people.get('bbox')
				if temp_bbox is not None:
					temp_bbox = self.xywh_to_x1y2x2y2(temp_bbox)
					temp_feature = self.extract_feature(temp_image, temp_bbox)
					cache_features.append(temp_feature)
					#print(temp_feature.shape)
					cache_ids.append(people.get('track_id'))

			right_flag_array = self._single_search_l2(query_feature, query_person_id, cache_features, cache_ids)
			all_flag += 1 
			right_cnt_array += right_flag_array
			now_acc = np.round(right_cnt_array / all_flag *100, 2)
			best_index = np.argmax(now_acc, axis=0)
			best_thresh, best_acc = self.test_thresh_list[best_index], now_acc[best_index]
			#print(now_acc)
			pbar.set_description('Query person  video:{} frame:{} track_id:{:>2} best_thresh_now:{} best_acc_now:{:>5}%  '.format(quert_person_video, query_person_frame, query_person_id, best_thresh, best_acc))
		pbar.close()
		accuracy = np.round(right_cnt_array / self.num_video *100, 2)

		return accuracy, self.num_video

if __name__ == '__main__':
	args = parseArgs()
	distance_thresh = args.dis_thresh
	model_path = '/export/home/cyh/posetrack/embedding/model_defense/model_weight.pth'
	data_path = '/export/home/zby/SiamFC/data/posetrack/images/val/'
	annotation_path = '/export/home/zby/SiamFC/data/posetrack/cyh_embedding/val/'
	num_video = 5000
	st = EmbeddingEvaluate(data_path, annotation_path, num_video)
	print('---------------------------------------')
	accuracy, num_video = st.eval()
	print('The real number of video is {} ,The accuracy is {}'.format(num_video, accuracy))







