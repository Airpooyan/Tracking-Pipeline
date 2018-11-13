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

class DateEncoder(json.JSONEncoder ):
	def default(self, obj):
		#print(obj,type(obj))
		if isinstance(obj,np.float32):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def data_preprocess(json_path, json_name):
	json_file = os.path.join(json_path,json_name)
	with open(json_file,'r') as f:
		images = json.load(f)['images']
	filenames_new = []
	frame_names = []
	for image in images:
		filename = image['file_name']
		frame_names.append(filename)
		filename = os.path.join('/export/home/zby/SiamFC-PyTorch/data/posetrack',filename)
		filenames_new.append(filename)
	frames = [cv2.imread(filename) for filename in filenames_new]
	return frames, frame_names

def track_test(gpu_id =[0,0,0], json_path='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'):
	json_files = os.listdir(json_path)
	tracker = Track_And_Detect(gpu_id=gpu_id)
	
	pose_vis_thresh = 0.2
	vis_flag = False

	
	predict_dict = dict()
	#random.shuffle(json_files)
	pbar = tqdm(range(len(json_files)))
	for json_name in json_files:
		video_json = {'annolist':[]}
		frames, frame_names = data_preprocess(json_path, json_name)
		video_name = json_name.replace('.json','')
		save_path = '/export/home/zby/SiamFC/data/test_result/{}.json'.format(video_name)
		pbar.set_description('Processing video {}'.format(video_name))
		pbar.update(1)
		for idx, frame in enumerate(frames):
			frame_name = frame_names[idx]
			if idx == 0:
				im_H,im_W,im_C = frame.shape
				if vis_flag:
					videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/{}.avi'.format(video_name),cv2.cv2.VideoWriter_fourcc('M','J','P','G'),10,(im_W,im_H))
				det_list = tracker.detect(frame)
				final_list = tracker.init_tracker(frame,det_list)
			else:	
				track_list = tracker.multi_track(frame)
				det_list = tracker.detect(frame)
				final_list = tracker.match_and_track(det_list, track_list, frame)
			image_dict = dict()

			annorect = []
			for det in final_list:
				point_list = []
				pose_position, pose_value, pose_heatmap = tracker.pose_detect(frame, det)
				#pose_position, pose_value = pose_position.tolist(), pose_value.tolist()
				for i, pose in enumerate(pose_position):
					score_i = pose_value[i]
					pose_id = match_list[i]
					if score_i >= pose_vis_thresh:
						point_list.append({'id':[pose_id],'x':[pose[0]],'y':[pose[1]],'score':[score_i]})
						if vis_flag:
							cv2.circle(frame,(int(pose[0]),int(pose[1])),10,(0,0,255),-1)
							cv2.putText(frame, str(i), (int(pose[0]+5),int(pose[1]+5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
				#print(det)
				point_dict = {'point':point_list}
				xmin,ymin,xmax,ymax,score,track_id = det
				annorect.append({'x1':[xmin],'x2':[xmax],'y1':[ymin],'y2':[ymax],'score':[score],'track_id':[track_id],'annopoints':[point_dict]})
				if vis_flag:
					cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
					cv2.putText(frame, 'id:'+str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
			image_dict['image'] = [{'name':frame_name}]
			image_dict['annorect'] = annorect
			video_json['annolist'].append(image_dict)
			if vis_flag:
				videoWriter.write(frame)
		with open(save_path,'w') as f:
			json.dump(video_json, f, cls=DateEncoder)
			#print('Tracking the {}th frame has taken {} seconds'.format(idx+1,end_time-start_time))
	pbar.close()


if __name__ == "__main__":
	#track(gpu_id=[0,0,0])
	json_path='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'
	track_test()
	# json_files = os.listdir(json_path)
	# for json_name in json_files:
		# frames = data_preprocess(json_path, json_name)
