# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

posetrack_classes = np.asarray(['__background__', 'pedestrian'])

pascal_classes = np.asarray(['__background__',
	'aeroplane', 'bicycle', 'bird', 'boat',
	'bottle', 'bus', 'car', 'cat', 'chair',
	'cow', 'diningtable', 'dog', 'horse',
	'motorbike', 'person', 'pottedplant',
	'sheep', 'sofa', 'train', 'tvmonitor'])

coco_classes = np.asarray(['__background__',
    'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']
     )

class Detector:
	def __init__(self, gpu_id=0, model_path = '/export/home/zby/SiamFC/models/res101/pascal_voc/faster_rcnn_1_25_4379.pth'):
		cfg_from_file('/export/home/zby/SiamFC/cfgs/res101.yml')
		cfg.USE_GPU_NMS = cfg.CUDA = True
		self.class_agnostic = False
		self.gpu_id = gpu_id
		self.model_path = model_path
		print('----------------------------------------')
		print('Detection: Initilizing detection network...')
		with torch.cuda.device(self.gpu_id):
			self._initialize_model()
			np.random.seed(cfg.RNG_SEED)
			self._initialize_data()
		zero_image = np.ones([600,1000,3])
		for i in range(3):
			self.detect(zero_image)
		print('Detection: Detection network has been initilized')
		
	def _initialize_model(self):
		fasterRCNN = resnet(posetrack_classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
		fasterRCNN.create_architecture()
		
		print("Detection: Loading checkpoint from %s" % (self.model_path))
		checkpoint = torch.load(self.model_path)
		if 'pooling_mode' in checkpoint.keys():
			cfg.POOLING_MODE = checkpoint['pooling_mode']
		#pprint.pprint(cfg)
		
		fasterRCNN.load_state_dict(checkpoint['model'])
		self.model = fasterRCNN.cuda()
		self.model.eval()
		
	def _initialize_data(self):
		# initilize the tensor holder here.
		im_data = torch.FloatTensor(1)
		im_info = torch.FloatTensor(1)
		num_boxes = torch.LongTensor(1)
		gt_boxes = torch.FloatTensor(1)

		# ship to cuda
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

		# make variable
		with torch.no_grad():
			self.im_data = Variable(im_data)
			self.im_info = Variable(im_info)
			self.num_boxes = Variable(num_boxes)
			self.gt_boxes = Variable(gt_boxes)

	def _get_image_blob(self, im):
		"""Converts an image into a network input.
		Arguments:
		im (ndarray): a color image in BGR order
		Returns:
		blob (ndarray): a data blob holding an image pyramid
		im_scale_factors (list): list of image scales (relative to im) used
		  in the image pyramid
		"""
		im_orig = im.astype(np.float32, copy=True)
		im_orig -= cfg.PIXEL_MEANS

		im_shape = im_orig.shape
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])

		processed_ims = []
		im_scale_factors = []

		for target_size in cfg.TEST.SCALES:
			im_scale = float(target_size) / float(im_size_min)
			# Prevent the biggest axis from being more than MAX_SIZE
			if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
				im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
			im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
							interpolation=cv2.INTER_LINEAR)
			im_scale_factors.append(im_scale)
			processed_ims.append(im)

		# Create a blob to hold the input images
		blob = im_list_to_blob(processed_ims)
		return blob, np.array(im_scale_factors)
	 
	def detect(self, im_in, nms_thresh=0.5, thresh=0.1):
		if len(im_in.shape) == 2:
			im_in = im_in[:,:,np.newaxis]
			im_in = np.concatenate((im_in,im_in,im_in), axis=2)
		# input must be bgr
		im = im_in

		blobs, im_scales = self._get_image_blob(im)
		assert len(im_scales) == 1, "Only single-image batch implemented"
		im_blob = blobs
		im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

		im_data_pt = torch.from_numpy(im_blob)
		im_data_pt = im_data_pt.permute(0, 3, 1, 2)
		im_info_pt = torch.from_numpy(im_info_np)

		with torch.cuda.device(self.gpu_id):
			self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
			self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
			self.gt_boxes.data.resize_(1, 1, 5).zero_()
			self.num_boxes.data.resize_(1).zero_()
			#print(self.im_info)
			#print(self.model)

			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label = self.model(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)
		
			scores = cls_prob.data
			boxes = rois.data[:, :, 1:5]

			if cfg.TEST.BBOX_REG:
				  # Apply bounding-box regression deltas
				box_deltas = bbox_pred.data
				if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
					if self.class_agnostic:			
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
									+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						box_deltas = box_deltas.view(1, -1, 4)
					else:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
									+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						#print(box_deltas.shape)
						box_deltas = box_deltas.view(1, -1, 4 * len(posetrack_classes))

				pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
				pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
			else:
				  # Simply repeat the boxes, once for each class
				pred_boxes = np.tile(boxes, (1, scores.shape[1]))

		pred_boxes /= im_scales[0]

		scores = scores.squeeze()
		pred_boxes = pred_boxes.squeeze()
		cls_dets = []
		for j in range(1, len(posetrack_classes)):
			inds = torch.nonzero(scores[:,j]>thresh).view(-1)
		  # if there is det
			if j>1:
				break
			if inds.numel() > 0:
				cls_scores = scores[:,j][inds]
				_, order = torch.sort(cls_scores, 0, True)
				if self.class_agnostic:
				  cls_boxes = pred_boxes[inds, :]
				else:
				  cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
				
				cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
				# cls_dets = torch.cat((cls_boxes, cls_scores), 1)
				cls_dets = cls_dets[order]
				keep = nms(cls_dets, nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
				cls_dets = cls_dets[keep.view(-1).long()]
				cls_dets = cls_dets.cpu().numpy()
		if len(cls_dets)==0:
			return cls_dets
		return cls_dets.tolist()

if __name__ == '__main__':
	detector = Detector()
	images = os.listdir('images')
	for image in images:
		im = cv2.imread('images/'+image)
		start = time.time()
		bboxes = detector.detect(im, nms_thresh=0.5, thresh=0.3)
		end = time.time()
		print('Detection has taken {} seconds'.format(end-start))
		if len(bboxes)>0:
			for det in bboxes:
				xmin,ymin,xmax,ymax,score = [int(x) for x in det]
				cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),1)
				cv2.putText(im,posetrack_classes[1],(xmin,ymin),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
		cv2.imwrite('data/result/'+image,im)
	#print(bbox)
