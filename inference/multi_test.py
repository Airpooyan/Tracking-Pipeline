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

def data_preprocess(json_path, json_name):
	#print('----------------------------------------')
	#print('Loading images....')
	json_file = os.path.join(json_path,json_name)
	with open(json_file,'r') as f:
		images = json.load(f)['images']
	filenames_new = []
	for image in images:
		filename = image['file_name']
		filename = os.path.join('/export/home/zby/SiamFC-PyTorch/data/posetrack',filename)
		filenames_new.append(filename)
	#image_path = json_file.replace('annotations','images').replace('.json','')
	#filenames = sorted(glob.glob(os.path.join(image_path, "*.jpg")),
	#	   key=lambda x: int(os.path.basename(x).split('.')[0]))
	#print(filenames  == filenames_new)
	
	frames = [cv2.imread(filename) for filename in filenames_new]
	#frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
	#print('Images has been loaded')
	return frames	

def track_test(gpu_id =[0,0,0], json_path='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'):
	json_files = os.listdir(json_path)
	tracker = Track_And_Detect(gpu_id=gpu_id)
	
	score_thresh = 4e-5
	predict_dict = dict()
	random.shuffle(json_files)
	for json_name in json_files:
		frames = data_preprocess(json_path, json_name)
		print('Processing video {}'.format(json_name))
		for idx, frame in enumerate(frames):
			if idx == 0:
				im_H,im_W,im_C = frame.shape
				videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/{}.avi'.format(json_name.replace('.json','')),cv2.cv2.VideoWriter_fourcc('M','J','P','G'),10,(im_W,im_H))
				det_list = tracker.detect(frame)
				final_list = tracker.init_tracker(frame,det_list)
				for det in final_list:
					#print(det)
					xmin,ymin,xmax,ymax,track_id = det
					cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
					cv2.putText(frame, 'id:'+str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)	
				videoWriter.write(frame)
				continue
			start_time = time.time()
			track_list = tracker.multi_track(frame)
			#print(track_list)
			det_list = tracker.detect(frame)
			final_list = tracker.match_and_track(det_list, track_list, frame)
			for bbox in final_list:
				xmin,ymin,xmax,ymax,track_id = bbox
				cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)            
				cv2.putText(frame, 'id:'+str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)
			# for bbox in det_list:
				# xmin,ymin,xmax,ymax,score = bbox
				# cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)            
				#cv2.putText(frame, 'score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
			#print(final_list)
			end_time = time.time()
			videoWriter.write(frame)
			print('Tracking the {}th frame has taken {} seconds'.format(idx+1,end_time-start_time))


if __name__ == "__main__":
	#track(gpu_id=[0,0,0])
	json_path='/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val'
	track_test()
	# json_files = os.listdir(json_path)
	# for json_name in json_files:
		# frames = data_preprocess(json_path, json_name)
