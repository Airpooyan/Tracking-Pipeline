import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import json

from fire import Fire
from tqdm import tqdm
import random

from detector import Detector
from pose_estimation import PoseNet
from tracker import SiamFCTracker
from embeddingNet import EmbeddingNet
from match import Matcher
from model.nms.nms_wrapper import nms

class Track_And_Detect:
	effective_track_thresh = 0.55
	effective_detection_thresh = 0.4
	
	effective_keypoints_thresh = 0.6
	effective_keypoints_number = 8
	
	iou_match_thresh = 0.5
	embedding_match_thresh = 2
	nms_thresh = 0.5
	oks_thresh = 0.8
	
	def __init__(self, gpu_id=[0,0,0,0],
							flag=[True, False, True, False],
							#track_model='/export/home/zby/SiamFC/models/output/siamfc_35.pth',
							track_model='/export/home/zby/SiamFC/models/output/siamfc_20.pth',
							detection_model='/export/home/zby/SiamFC/models/res101_old/pascal_voc/faster_rcnn_1_25_4379.pth',
							pose_model='/export/home/zby/SiamFC/data/models/final_new.pth.tar',
							embedding_model='/export/home/zby/SiamFC/models/embedding_model.pth'):
		if flag[0]:
			self.tracker = SiamFCTracker(gpu_id[0], track_model)#input RGB
		if flag[1]:
			self.detector = Detector(gpu_id[1], detection_model)#input BGR
		if flag[2]:
			self.posenet = PoseNet(gpu_id[2], pose_model)#input BGR
		if flag[3]:
			self.embedder = EmbeddingNet(gpu_id[3], embedding_model)
		#self.tracker = SiamFCTracker(gpu_id[0], track_model)
		self.matcher = Matcher()
		print('----------------------------------------')
		
	#initialize the first frame of this video
	def init_tracker(self, frame, bbox_list):
		self.new_id_flag=0
		self.track_id_dict= dict()
		#self.tracker.clear_data()
		#conver bgr(opencv) to rgb
		rgb_frame = frame[:,:,::-1]
		bbox_list, keypoint_list = self.oks_filter(bbox_list, frame)
		#print(bbox_list)
		for bbox in bbox_list:
			self.create_id(frame, rgb_frame, bbox)
		bbox_list=[]
		#pose_list=[]
		for id,item in self.track_id_dict.items():
			bbox = item['bbox_and_score'] + [id]
			bbox_list.append(bbox)
		#	pose_position, pose_value, pose_heatmap = self.pose_detect(frame, bbox)
		#	pose_info = np.hstack(pose_postion, pose_value)
		return bbox_list
		
	def oks_filter(self, det_list, frame):
		keypoint_list = []
		for bbox in det_list:
			center, scale = self.posenet.x1y1x2y2_to_cs(bbox[0:4])
			area = np.prod(scale*200,1)
			pred = np.zeros((15,3), dtype=np.float32)
			pose_positions, pose_vals, pose_heatmaps = self.pose_detect(frame, bbox)
			#print(pose_vals)
			#posa_vals = np.expand_dims(pose_vals, axis=1)
			pred[:,0:2] = pose_positions
			pred[:,2] = pose_vals
			score_all,valid_num = 0, 0
			for i in range(15):
				score_i = pose_vals[i]
				if score_i >= 0.2:
					score_all += score_i
					valid_num += 1
			if valid_num!=0:
				new_score = score_all/valid_num *bbox[4]
			else:
				new_score = 0
			keypoint_dict={'score':new_score, 'area':area, 'keypoints':pred}
			keypoint_list.append(keypoint_dict)
		keep = self.matcher.oks_nms(keypoint_list, thresh= self.oks_thresh)
		new_det_list = [det_list[i] for i in keep]
		new_keypoint_list = [keypoint_list[i] for i in keep]
		return new_det_list, new_keypoint_list
		
	def create_id(self, frame, rgb_frame, bbox):
		score = bbox[4]
		bbox = bbox[0:4]
		track_id = self.new_id_flag
		feature = self.embedding(frame, bbox)
		#self.track_id_dict[track_id]={'bbox_and_score':bbox+[score]}
		self.track_id_dict[track_id]={'bbox_and_score':bbox+[score],'feature':feature, 'frame_flag':1, 'exist':True}
		#self.track_id_dict[track_id]={'bbox_and_score':bbox+[score],'feature':[feature], 'frame_flag':1, 'exist':True}
		#pose_position, pose_value, pose_heatmap = self.pose_detect(frame, bbox)
		#self.update_tracker(rgb_frame, bbox, track_id)
		#print('Track id {} has been initinized'.format(track_id))
		self.new_id_flag += 1
		
	def update_id(self, frame, rgb_frame, det_bbox, track_id):
		bbox, score = det_bbox[0:4], det_bbox[4]
		feature = np.array(self.embedding(frame, bbox))
		former_track_dict = self.track_id_dict[track_id]
		former_frame_flag, former_feature = former_track_dict['frame_flag'], np.array(former_track_dict['feature'])
		now_frame_flag = former_frame_flag+1
		#calculate the average feature
		#now_feature = ((former_feature*former_frame_flag+feature)/now_frame_flag).tolist()
		#former_feature = former_feature.tolist()
		now_feature = feature.tolist()
		#former_feature.append(now_feature)
		self.track_id_dict[track_id]={'bbox_and_score':det_bbox,'feature':now_feature, 'frame_flag':now_frame_flag, 'exist':True}
		#pose_position, pose_value, pose_heatmap = self.pose_detect(frame, bbox)
		#self.update_tracker(rgb_frame, bbox, track_id)
		
	def multi_track(self, frame):
		rgb_frame = frame[:,:,::-1]
		bbox_list = []
		for id in self.track_id_dict:
			if self.track_id_dict[id]['exist'] == False:
				continue
			bbox, score = self.tracker.track_id(rgb_frame, id)
			bbox_list.append(bbox +[score] +[id])
			self.track_id_dict[id]['bbox'] = bbox
		return bbox_list
		
	def match_and_track_embedding(self, detections, track_list, frame):
		# print(self.track_id_dict.keys())
		# exist_flag = []
		# for id in self.track_id_dict:
			# exist_flag.append(self.track_id_dict[id]['exist'])
		# print(exist_flag)
		rgb_frame = frame[:,:,::-1]
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, track_list, iou_threshold = self.iou_match_thresh)
		
		has_tracked_id = set()
		for match in matches:
			det_index, track_index = match
			has_tracked_id.add(track_index)
			det_bbox = detections[det_index]
			update_id = track_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			
		#get feature unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		#get feature for unmatched_trackers
		database_feature_list = []
		for database_id in self.track_id_dict:
			if database_id in has_tracked_id:
				continue
			database_id_feature = self.track_id_dict[database_id]['feature'] + [database_id]
			database_feature_list.append(database_id_feature)
			
		# for delete_index in unmatched_trackers:
			# track_bbox = track_list[delete_index]
			# track_score, delete_id = track_bbox[4], track_bbox[5]
			# delete_id_feature = self.track_id_dict[delete_id]['feature'] + [delete_id]
			# track_feature_list.append(delete_id_feature)
			
		#match the detection and tracklist
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								database_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = database_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#change status for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			self.track_id_dict[delete_id]['exist'] = False
			
		#delete unuseful index for unmatched_trackers
		# for delete_index in embedding_unmatched_trackers:
			# delete_id = track_feature_list[delete_index][2048]
			# del self.track_id_dict[delete_id]
			# self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
		
	def match_detection_embedding(self, detections, frame):
		rgb_frame = frame[:,:,::-1]
		#get detection feature
		det_feature_list =[]
		for det_bbox in detections:
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		#get feature for former trackers
		database_feature_list = []
		for database_id in self.track_id_dict:
			database_id_feature = self.track_id_dict[database_id]['feature'] + [database_id]
			database_feature_list.append(database_id_feature)
			
		#match the detection and tracklist
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								database_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = database_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = database_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list

	def match_detection_iou(self, detections, frame):
		rgb_frame = frame[:,:,::-1]
			
		#get feature for former trackers
		database_bbox_list = []
		for database_id in self.track_id_dict:
			database_id_bbox = self.track_id_dict[database_id]['bbox_and_score']+[database_id]
			database_bbox_list.append(database_id_bbox)
		
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, database_bbox_list, iou_threshold = self.iou_match_thresh)
		
		#update the matched trackers with detection bbox
		for match in matches:
			det_index, track_index = match
			det_bbox = detections[det_index]
			update_id = database_bbox_list[track_index][5]
			self.track_id_dict[update_id]['bbox_and_score'] = det_bbox[0:5]
			
		#create new index for unmatched_detections
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in unmatched_trackers:
			delete_id = database_bbox_list[delete_index][5]
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
			
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
		
	def match_detection_iou_embedding(self, detections, frame):
		rgb_frame = frame[:,:,::-1]
			
		#final_bbox = []
		detections, keypoint_list = self.oks_filter(detections, frame)
		#get feature for former trackers
		database_bbox_list = []
		for database_id in self.track_id_dict:
			database_id_bbox = self.track_id_dict[database_id]['bbox_and_score']+[database_id]
			database_bbox_list.append(database_id_bbox)
		
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, database_bbox_list, iou_threshold = self.iou_match_thresh)
		
		#update the matched trackers with detection bbox
		for match in matches:
			det_index, track_index = match
			det_bbox = detections[det_index]
			update_id = database_bbox_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			#final_bbox.append(det_bbox+[update_id])
			
		#create new index for unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		track_feature_list = []
		for delete_index in unmatched_trackers:
			track_bbox = database_bbox_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			delete_id_feature = self.track_id_dict[delete_id]['feature'] + [delete_id]
			track_feature_list.append(delete_id_feature)
			
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = track_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			#self.tracker.delete_id(delete_id)
			
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		return bbox_list
		
	def match_detection_tracking_iou_embedding(self, detections, track_list, frame):
		rgb_frame = frame[:,:,::-1]
			
		#print(detections)
		for track in track_list:
			track_score = track[4]
			if track_score >= self.effective_track_thresh:
				detections.append(track[0:5])
		#print(detections)
		detections, keypoint_list = self.oks_filter(detections, frame)
		#get feature for former trackers
		database_bbox_list = []
		for database_id in self.track_id_dict:
			database_id_bbox = self.track_id_dict[database_id]['bbox_and_score']+[database_id]
			database_bbox_list.append(database_id_bbox)
		
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, database_bbox_list, iou_threshold = self.iou_match_thresh)
		
		#update the matched trackers with detection bbox
		for match in matches:
			det_index, track_index = match
			det_bbox = detections[det_index]
			update_id = database_bbox_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			#final_bbox.append(det_bbox+[update_id])
			
		#create new index for unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		track_feature_list = []
		for delete_index in unmatched_trackers:
			track_bbox = database_bbox_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			delete_id_feature = self.track_id_dict[delete_id]['feature'] + [delete_id]
			track_feature_list.append(delete_id_feature)
			
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = track_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
			
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		return bbox_list
		
	def match_and_track_embedding_no_database(self, detections, track_list, frame):
		# print(self.track_id_dict.keys())
		rgb_frame = frame[:,:,::-1]
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, track_list, iou_threshold = self.iou_match_thresh)
		
		has_tracked_id = set()
		for match in matches:
			det_index, track_index = match
			has_tracked_id.add(track_index)
			det_bbox = detections[det_index]
			update_id = track_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			
		#get feature unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		#get feature for unmatched_trackers
		track_feature_list = []
		for delete_index in unmatched_trackers:
			track_bbox = track_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			delete_id_feature = self.track_id_dict[delete_id]['feature'] + [delete_id]
			track_feature_list.append(delete_id_feature)
		#match the detection and tracklist
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = track_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
		
	def match_and_track_embedding_temporal_database(self, detections, track_list, frame):
		# print(self.track_id_dict.keys())
		rgb_frame = frame[:,:,::-1]
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, track_list, iou_threshold = self.iou_match_thresh)
		
		has_tracked_id = set()
		for match in matches:
			det_index, track_index = match
			has_tracked_id.add(track_index)
			det_bbox = detections[det_index]
			update_id = track_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			
		#get feature unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		#get feature for unmatched_trackers
		track_feature_list = []
		for delete_index in unmatched_trackers:
			track_bbox = track_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			delete_id_features = self.track_id_dict[delete_id]['feature']
			for delete_id_feature in delete_id_features:
				track_feature_list.append(delete_id_feature + [delete_id])
		#match the detection and tracklist
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		update_ids=dict()
		for match in embedding_matches:
			det_index, track_index = match
			update_id = track_feature_list[track_index][2048]
			if not update_id in update_ids:
				update_ids[update_id] = det_index
			else:
				score_former = det_feature_list[update_ids[update_id]][-1]
				score_now = det_feature_list[det_index][-1]
				if score_now>=score_former:
					update_ids[update_id] = det_index
				
		for update_id,det_index in update_ids.items():
			det_bbox = det_feature_list[det_index][2048:]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
				
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			if delete_id not in update_ids and delete_id in self.track_id_dict:
				del self.track_id_dict[delete_id]
				self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
	
	def match_and_track_embedding_no_database_nms(self, detections, track_list, frame):
		# print(self.track_id_dict.keys())
		rgb_frame = frame[:,:,::-1]
		all_bbox = []
		for det in detections:
			all_bbox.append(det)
		for track in track_list:
			if not track[4] >= self.effective_track_thresh:
				continue
			all_bbox.append(track[0:4]+[track[5]])
		all_bbox = np.array(all_bbox)
		#print(all_bbox)
		keep = self.matcher.nms(all_bbox, self.nms_thresh)
		keep_bboxes = all_bbox[keep].tolist()
		#print(len(all_bbox), len(keep_bboxes))
		#print(keep_bboxes)
			
		#get feature unmatched_detections
		det_feature_list =[]
		for keep_bbox in keep_bboxes:
			det_id_feature = self.embedding(frame, keep_bbox) + keep_bbox
			det_feature_list.append(det_id_feature)
			
		#get feature for unmatched_trackers
		track_feature_list = []
		for former_id in self.track_id_dict:
			former_id_feature = self.track_id_dict[former_id]['feature'] + [former_id]
			track_feature_list.append(former_id_feature)
		#match the detection and tracklist
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = track_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
		
	def match_and_track_iou(self, detections, track_list, frame):
		rgb_frame = frame[:,:,::-1]
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, track_list, iou_threshold = self.iou_match_thresh)
		
		#update the matched trackers with detection bbox
		for match in matches:
			det_index, track_index = match
			det_bbox = detections[det_index]
			update_id = track_list[track_index][5]
			#print('id {} has been updated'.format(update_id))
			self.track_id_dict[update_id]['bbox_and_score'] = det_bbox[0:5]
			self.update_tracker(rgb_frame, det_bbox, update_id)
			
		#create new index for unmatched_detections
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				#print('add id{} now'.format(self.new_id_flag))
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in unmatched_trackers:
			track_bbox = track_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			#pose_position, pose_value, pose_heatmap = self.pose_detect(frame, track_bbox)
			# if track_score >= self.effective_track_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				# self.track_id_dict[delete_id]['bbox_and_score']=track_bbox[0:5]
			# else:
				# #print('delete id{} now'.format(delete_id))
			del self.track_id_dict[delete_id]
			self.tracker.delete_id(delete_id)
		
		bbox_list = []
		for id,item in self.track_id_dict.items():
			bbox_list.append(item['bbox_and_score']+[id])
		#print(bbox_list)
		return bbox_list
		
	#bbox must be format of x1y1x2y2
	def update_tracker(self, rgb_frame, bbox, track_id):
		self.tracker.update_data_dict(rgb_frame, bbox, track_id)
	
	def embedding(self, frame, bbox):
		start = time.time()
		#print(bbox)
		feature = self.embedder.get_feature(frame,bbox)
		#print('Embedding takes {}s'.format(time.time()-start))
		return feature
		
	def detect(self, im):
		return self.detector.detect(im, self.nms_thresh, 0.5)
	
	def pose_detect(self, im, bbox):
		return self.posenet.detect_pose(im, bbox)

def data_process(json_file_path ='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'):
	print('----------------------------------------')
	print('Loading images....')
	# load videos
	json_files = os.listdir(json_file_path)
	random_number = random.randint(0,len(json_files))
	json_name = json_files[random_number]
	#json_name = '002277_mpii_test.json'
	json_file = os.path.join(json_file_path,json_name)
	with open(json_file,'r') as f:
		annotation = json.load(f)['annotations']
	bbox_dict = dict()
	for anno in annotation:
		track_id = anno['track_id']
		frame_id = anno['image_id'] % 1000
		if not 'bbox' in anno or frame_id!=0:
			continue
		bbox = anno['bbox']        
		if bbox[2]==0 or bbox[3]==0:
			continue
		bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
		if not track_id in bbox_dict:
			bbox_dict[track_id] = {'bbox':bbox,'frame_id':frame_id} 
	image_path = json_file.replace('annotations','images').replace('.json','')
	filenames = sorted(glob.glob(os.path.join(image_path, "*.jpg")),
		   key=lambda x: int(os.path.basename(x).split('.')[0]))
	frames = [cv2.imread(filename) for filename in filenames]
	#frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
	print('Images has been loaded')
	im_H,im_W,im_C = frames[0].shape
	videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/{}.avi'.format(json_name.replace('.json','')),cv2.cv2.VideoWriter_fourcc('M','J','P','G'),10,(im_W,im_H))
	return videoWriter, frames, bbox_dict		

if __name__ == "__main__":
    #track_and_detect(model_path='/export/home/zby/SiamFC/models/output/siamfc_35.pth')
	mytrack = Track_And_Detect(gpu_id=[0,0,0])
	#track_(0)
	
