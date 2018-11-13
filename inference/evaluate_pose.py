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

from track_and_detect import Track_And_Detect, data_process

'''
For posetrack dataset, the output keypoints is as follow 
"keypoints": {
	0: "nose",
	1: "head_bottom",
	2: "head_top",
	3: "left_shoulder",
	4: "right_shoulder",
	5: "left_elbow",
	6: "right_elbow",
	7: "left_wrist",
	8: "right_wrist",
	9: "left_hip",
	10: "right_hip",
	11: "left_knee",
	12: "right_knee",
	13: "left_ankle",
	14: "right_ankle"
}
For competition
"keypoints": {
  0: "right_ankle",
  1: "right_knee",
  2: "right_hip",
  3: "left_hip",
  4: "left_knee",
  5: "left_ankle",
  6: "right_wrist",
  7: "right_elbow",
  8: "right_shoulder",
  9: "left_shoulder",
  10: "left_elbow",
  11: "left_wrist",
  12: "neck",
  13: "nose",
  14: "head_top",
}
'''
match_list=[13,12,14,9,8,10,7,11,6,3,2,4,1,5,0]
with_gt = False

class DateEncoder(json.JSONEncoder ):
	def default(self, obj):
		#print(obj,type(obj))
		if isinstance(obj,np.float32):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def data_preprocess(json_dict, video_name):
	annotations = json_dict[video_name]['annotations']
	images = json_dict[video_name]['images']
	image_dict = dict()
	for image in images:
		filename = image['file_name']
		image_dict[filename]=[]
	for anno in annotations:
		if not 'bbox' in anno:
			continue
		xmin,ymin,w,h = anno['bbox']
		filename = anno['file_name']
		if w<=0 or h<=0 or w>=1200 or h>=1200:
			continue
		xmax,ymax = xmin+w, ymin+h
		image_dict[filename].append([xmin,ymin,xmax,ymax])
	#print(image_dict)
	return image_dict

def data_preprocess(json_dict, video_name):
	annotations = json_dict[video_name]['annotations']
	images = json_dict[video_name]['images']
	image_dict = dict()
	for image in images:
		filename = image['file_name']
		image_dict[filename]=[]
	for anno in annotations:
		if not 'bbox' in anno:
			continue
		xmin,ymin,w,h = anno['bbox']
		filename = anno['file_name']
		if w<=0 or h<=0 or w>=1200 or h>=1200:
			continue
		xmax,ymax = xmin+w, ymin+h
		image_dict[filename].append([xmin,ymin,xmax,ymax])
	#print(image_dict)
	return image_dict

def write_to_json(gpu_id =[0,0,0], gt_path='/export/home/zby/SiamFC/data/posetrack/posetrack_val.json', det_path='/export/home/zby/SiamFC/data/posetrack/tf_detection_result.json'):
	if with_gt:
		with open(gt_path,'r') as f:
			json_dict = json.load(f)
	else:
		with open(det_path,'r') as f:
			json_dict = json.load(f)		
	tracker = Track_And_Detect(gpu_id=gpu_id,flag = [False,False,True,False])
	
	pose_vis_thresh = 0
	vis_flag = False
	detection_score_thresh = 0.5
	
	predict_dict = dict()
	#random.shuffle(json_files)
	pbar = tqdm(range(len(json_dict)))
	for video_name in json_dict:
		video_json = {'annolist':[]}
		if with_gt:
			gt_dict = data_preprocess(json_dict, video_name)
		else:
			frame_dict = json_dict[video_name]
		save_path = '/export/home/zby/SiamFC/data/posetrack/evaluate_pose_embeddingPose/{}.json'.format(video_name)
		#pbar.set_description('Processing video {}'.format(video_name))
		pbar.update(1)
		idx=0
		for frame_name in sorted(frame_dict.keys()):
			#print(frame_name)
			start = time.time()
			frame_path = os.path.join('/export/home/zby/SiamFC/data/posetrack',frame_name)
			frame = cv2.imread(frame_path)
			bbox_list = frame_dict[frame_name]
			image_dict = dict()
			annorect = []
			det_list = []
			for bbox in bbox_list:
				if bbox[4] >= detection_score_thresh:
					det_list.append(bbox)
			if idx==0 and vis_flag == True:
				im_H,im_W,im_C = frame.shape
				if vis_flag:
					videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/{}.avi'.format(video_name),cv2.cv2.VideoWriter_fourcc('M','J','P','G'),10,(im_W,im_H))
			for det in det_list:
				point_list = []
				pose_position, pose_value, pose_heatmap = tracker.pose_detect(frame, det)
				#pose_position, pose_value = pose_position.tolist(), pose_value.tolist()
				for i, pose in enumerate(pose_position):
					score_i = pose_value[i]
					pose_id = match_list[i]
					#if score_i >= pose_vis_thresh:
					point_list.append({'id':[pose_id],'x':[pose[0]],'y':[pose[1]],'score':[score_i]})
					if vis_flag:
							cv2.circle(frame,(int(pose[0]),int(pose[1])),10,(0,0,255),-1)
							cv2.putText(frame, str(i), (int(pose[0]+5),int(pose[1]+5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
				#print(det)
				point_dict = {'point':point_list}
				xmin,ymin,xmax,ymax,score = det
				#new_score = score * sum(pose_value)/15
				track_id = 0
				annorect.append({'x1':[xmin],'x2':[xmax],'y1':[ymin],'y2':[ymax],'score':[score],'track_id':[0],'annopoints':[point_dict]})
				if vis_flag:
					cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
					cv2.putText(frame, 'id:'+str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
			image_dict['image'] = [{'name':frame_name}]
			image_dict['annorect'] = annorect
			video_json['annolist'].append(image_dict)
			if vis_flag:
				videoWriter.write(frame)
			idx += 1
			pbar.set_description('Processing video {}: process {} takes {:.3f} seconds'.format(video_name, frame_name, time.time()-start))
		with open(save_path,'w') as f:
			json.dump(video_json, f, cls=DateEncoder)
			#print('Tracking the {}th frame has taken {} seconds'.format(idx+1,end_time-start_time))
	pbar.close()



if __name__ == "__main__":
	#track(gpu_id=[0,0,0])
	json_path='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'
	write_to_json()
	# json_files = os.listdir(json_path)
	# for json_name in json_files:
		# frames = data_preprocess(json_path, json_name)
