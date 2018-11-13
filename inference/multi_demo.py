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

def data_process(json_file_path ='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'):
	print('----------------------------------------')
	print('Loading images....')
	# load videos
	json_files = os.listdir(json_file_path)
	random_number = random.randint(0,len(json_files))
	json_name = json_files[random_number]
	json_name = '013534_mpii_test.json'
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
	return videoWriter, frames, bbox_dict, json_name	


def track(gpu_id =0):
	videoWriter, frames, bbox_dict, json_name = data_process()
	tracker = Track_And_Detect(gpu_id=0)

	# starting tracking
	#print(bbox_dict)
	print('First frame has totally {} boxes for tracking'.format(len(bbox_dict)))
	bbox_list = []
	for key, item in bbox_dict.items():
		x,y,w,h = item['bbox']
		bbox_list.append([x, y, x+w, y+h, 1])
	score_thresh = 0.01
	predict_dict = dict()
	tracker.init_tracker(frames[0],bbox_list)
	
	pbar = tqdm(range(len(frames)))
	for idx, frame in enumerate(frames):
		flag = False
		det_id = []
		start_time = time.time()
		bbox_list = tracker.multi_track(frame)
		#print(bbox_list)
		end_time = time.time()
		pbar.update(1)
		pbar.set_description('Processing video {} : Tracking the {:>3}th frame has taken {:.3f} seconds'.format(json_name, idx+1,end_time-start_time))
		for bbox in bbox_list:
			score,track_id = bbox[4], bbox[5]
			if score < score_thresh:
				flag = True
				det_id.append(track_id)
		if flag == True:
			pbar.set_description('Track id {} in {}th frame has failed, need detection to revise'.format(det_id, idx+1))
			start_time = time.time()
			det_boxes = tracker.detect(frame)
			tracker.match_id(det_boxes)
			det_poses = []
			for det_box in det_boxes:
				pose_position, pose_val, pose_heatmap = tracker.pose_detect(frame, det_box)
				#print(det_box, pose_position)
				det_poses.append(pose_position)
			#print(det_poses)
			pbar.set_description('Detect the {}th frame has taken {} seconds'.format(idx+1,time.time()-start_time))

		for bbox in bbox_list:
			xmin,ymin,xmax,ymax,score,track_id = bbox
			score = np.round(score, 3)
			if score< score_thresh:
				cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
				cv2.putText(frame, 'id:'+str(track_id)+' score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
			else:                  
				cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
				cv2.putText(frame, 'id:'+str(track_id)+' score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)				
		if flag == True:
			for i,bbox in enumerate(det_boxes):
				det_pose = det_poses[i]
				for position in det_pose:
					x,y = position
					cv2.circle(frame,(int(x),int(y)),10,(0,0,255),-1)
				xmin,ymin,xmax,ymax,score = bbox
				cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,255), 2)
				cv2.putText(frame, 'detection', (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
		videoWriter.write(frame)
	pbar.close()


if __name__ == "__main__":
    #track_and_detect(model_path='/export/home/zby/SiamFC/models/output/siamfc_35.pth')
	#mytrack = Track_And_Detect(gpu_id=0)
	track(gpu_id=0)
	# tracker = Track_And_Detect(gpu_id=[0,0,0])
	# im_path = '/export/home/zby/SiamFC/data/demo_images/13.jpg'
	# im = cv2.imread(im_path)
	# bbox = [75,56,360,380,0.99,8]
	# preds, maxvals, heatmaps = tracker.pose_detect(im, bbox)
	# for pred in preds:
		# x,y = pred
		# cv2.circle(im,(int(x),int(y)), 5, (0,0,255), -1)
	# cv2.imwrite('result.jpg',im)