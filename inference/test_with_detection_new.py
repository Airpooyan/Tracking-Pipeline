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

from track_and_detect_new import Track_And_Detect

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

def parseArgs():
	parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
	parser.add_argument("-t", "--detection_thresh",dest = 'det_thresh',required=False, default=0.4, type= float)
	parser.add_argument("-p", "--pos_thresh",dest = 'pose_thresh',required=False, default=0, type= float)
	parser.add_argument("-v", "--vis_flag",dest = 'vis_flag',required=False, default=False, type= bool)
	return parser.parse_args()

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

def track_test(args, gpu_id=0, json_path='/export/home/zby/SiamFC/data/posetrack/tf_detection_result.json'):
	pose_vis_thresh = args.pose_thresh
	detection_score_thresh = args.det_thresh
	vis_flag = args.vis_flag
	#save_dir = '/export/home/zby/SiamFC/data/posetrack/test_with_detection_result_{}'.format(detection_score_thresh)
	save_dir = '/export/home/zby/SiamFC/data/posetrack/detection_tracking_oks_newEmbedding_decrease_final'
	print('----------------------------------------------------------------------------------')
	print('Detection_score_thresh: {}    Vis_flag: {}'.format(detection_score_thresh, vis_flag))
	print('Detection results is set as {}'.format(json_path))
	print('Results will be saved to {}'.format(save_dir))
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	
	with open(json_path,'r') as f:
		bbox_dict = json.load(f)
	tracker = Track_And_Detect(gpu_id=gpu_id)
	gt_json_path = '/export/home/zby/SiamFC/data/posetrack/posetrack_val_2017.json'
	with open(gt_json_path,'r') as f:
		gt_dict = json.load(f)
	video_keys = gt_dict.keys()
	#video_keys = bbox_dict.keys()
	predict_dict = dict()
	#random.shuffle(json_files)
	pbar = tqdm(range(len(video_keys)))
	for video_name in video_keys:
		# if video_name != '007496_mpii_test':
			# continue
		frame_dict = bbox_dict[video_name]
		video_json = {'annolist':[]}
		save_path = os.path.join(save_dir, video_name+'.json')
		idx =0
		for frame_name in sorted(frame_dict.keys()):
			start = time.time()
			frame_path = os.path.join('/export/home/zby/SiamFC/data/posetrack',frame_name)
			frame = cv2.imread(frame_path)
			bbox_list = frame_dict[frame_name]
			det_list = []
			for bbox in bbox_list:
				if bbox[4] >= detection_score_thresh:
					det_list.append(bbox)
			if idx == 0:
				im_H,im_W,im_C = frame.shape
				if vis_flag:
					fourcc = cv2.cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
					videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/detection_tracking_oks_newEmbedding_decrease_final_{}.mp4'.format(video_name),fourcc,10,(im_W,im_H))
				final_list = tracker.init_tracker(frame,det_list)
				# for det in det_list:
					# xmin, ymin, xmax, ymax, score = det
					# cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2) 
				# for det in final_list:
					# xmin, ymin, xmax, ymax, score, track_id = det
					# cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2) 	
				# print(len(det_list),len(final_list))
				# if (len(det_list)!=len(final_list)):
					# cv2.imwrite('/export/home/zby/SiamFC/data/result/'+video_name+'_pred.jpg',frame)
			else:	
				track_list = tracker.multi_track(frame)
				
				#final_list = tracker.match_detection_iou(det_list, frame)
				#final_list = tracker.match_detection_embedding(det_list, frame)
				#final_list = tracker.match_detection_iou_embedding(det_list, frame)
				final_list = tracker.match_detection_tracking_oks_embedding(det_list, track_list, frame)
				#final_list = tracker.match_detection_tracking_oks_iou_embedding(det_list, track_list, frame)
				#final_list = tracker.match_detection_tracking_iou_embedding(det_list, track_list, frame)

			image_dict = dict()
			annorect = []
			for det in final_list:
				point_list = []
				pose_position, pose_value, pose_heatmap = tracker.pose_detect(frame, det)
				for i, pose in enumerate(pose_position):
					score_i = pose_value[i]
					pose_id = match_list[i]
					point_list.append({'id':[pose_id],'x':[pose[0]],'y':[pose[1]],'score':[score_i]})
				point_dict = {'point':point_list}
				xmin,ymin,xmax,ymax,score,track_id = det
				annorect.append({'x1':[xmin],'x2':[xmax],'y1':[ymin],'y2':[ymax],'score':[score],'track_id':[track_id],'annopoints':[point_dict]})
			image_dict['image'] = [{'name':frame_name}]
			image_dict['annorect'] = annorect
			video_json['annolist'].append(image_dict)
			idx += 1
			pbar.set_description('Processing video {}: process {} takes {:.3f} seconds'.format(video_name, frame_name, time.time()-start))
			
			if vis_flag:
				for anno in annorect:
					xmin, ymin, xmax, ymax, score, track_id = anno['x1'][0], anno['y1'][0], anno['x2'][0], anno['y2'][0], anno['score'][0], anno['track_id'][0]
					cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
					cv2.putText(frame, 'id:'+str(track_id)+'score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
					point_list = anno['annopoints'][0]['point']
					for point in point_list:
						point_id, x, y, score = point['id'][0], point['x'][0], point['y'][0], point['score'][0]
						cv2.circle(frame,(int(x),int(y)),10,(0,0,255),-1)
						cv2.putText(frame, str(point_id), (int(x+5),int(y+5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
				videoWriter.write(frame)
		pbar.update(1)
		with open(save_path,'w') as f:
			json.dump(video_json, f, cls=DateEncoder)
	pbar.close()


if __name__ == "__main__":
	args = parseArgs()
	track_test(args=args)
