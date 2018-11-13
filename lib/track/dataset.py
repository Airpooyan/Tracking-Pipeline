import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from pose.core.config import update_config

from .config import config

class ImagnetVIDDataset(Dataset):
	def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
		self.video_names = video_names
		self.data_dir = data_dir
		self.z_transforms = z_transforms
		self.x_transforms = x_transforms
		meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
		self.meta_data = pickle.load(open(meta_data_path, 'rb'))
		self.meta_data = {x[0]:x[1] for x in self.meta_data}
		self.pose_transform = transforms.Compose([transforms.ToTensor(),
											 transforms.Normalize(mean = [0.485,0.456,0.406],
																  std = [0.229,0.224,0.225])
											])		
		self.tensor_transform = transforms.Compose([transforms.ToTensor()])
		# filter traj len less than 2
		for key in self.meta_data.keys():
			#print(key)
			trajs = self.meta_data[key]
			#print(trajs)
			for trkid in list(trajs.keys()):
				if len(trajs[trkid]) < 2:
					del trajs[trkid]

		self.txn = db.begin(write=False)
		print(len(self.video_names))
		self.num = len(self.video_names) if config.num_per_epoch is None or not training\
				else config.num_per_epoch

	def imread(self, path):
		key = hashlib.md5(path.encode()).digest()
		#print(key)
		img_buffer = self.txn.get(key)
		#print(img_buffer)
		img_buffer = np.frombuffer(img_buffer, np.uint8)
		img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
		return img

	def __getitem__(self, idx):
		idx = idx % len(self.video_names)
		video = self.video_names[idx]
		trajs = self.meta_data[video]
		# sample one trajs
		trkid = np.random.choice(list(trajs.keys()))
		traj = trajs[trkid]
		assert len(traj) > 1, "video_name: {}".format(video)
		# sample exemplar
		exemplar_idx = np.random.choice(list(range(len(traj))))
		exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.jpg".format(trkid))
		#print(exemplar_name)
		exemplar_img = self.imread(exemplar_name)
		exemplar_img = self.z_transforms(exemplar_img)
		exemplar_pose_im = cv2.resize(exemplar_img, (288,384), interpolation=cv2.INTER_LINEAR)
		exemplar_pose_im = self.pose_transform(exemplar_pose_im)
		exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
		exemplar_img = self.tensor_transform(exemplar_img)
		
		# sample instance
		low_idx = max(0, exemplar_idx - config.frame_range)
		up_idx = min(len(traj), exemplar_idx + config.frame_range)
		instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx])
		instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.jpg".format(trkid))
		instance_img = self.imread(instance_name)
		#print(instance_img.shape)
		instance_img = self.x_transforms(instance_img)
		#print(instance_img.shape)
		instance_pose_im = cv2.resize(instance_img, (288,384), interpolation=cv2.INTER_LINEAR)
		instance_pose_im = self.pose_transform(instance_pose_im)
		instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
		instance_img = self.tensor_transform(instance_img)
		
		
		return exemplar_img, instance_img, exemplar_pose_im, instance_pose_im

	def __len__(self):
		return self.num
