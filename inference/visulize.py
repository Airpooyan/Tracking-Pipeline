import json
import os
import argparse
from tqdm import tqdm
import cv2
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
'''
[`nose`, `upper_neck`, `head_top`, `left_shoulder`,
`right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`,
`left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`,
`right_ankle`].
'''

def parseArgs():

    parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
    parser.add_argument("-n", "--number",dest = 'pose_number',required=False, default=8, type= int)
    parser.add_argument("-t", "--thresh",dest = 'pose_thresh',required=False, default=0.1, type=float)
    return parser.parse_args()

def process_gt(video_name, json_dict):
	annotations = json_dict[video_name]['annotations']
	images = json_dict[video_name]['images']
	image_dict = dict()
	for image in images:
		filename = image['file_name']
		image_dict[filename]={'bbox':[],'keypoint':[]}
	for anno in annotations:
		if not 'bbox' in anno:
			continue
		filename = anno['file_name']
		track_id = anno['track_id']
		
		xmin,ymin,w,h = anno['bbox']
		if w<=0 or h<=0 or w>=1200 or h>=1200:
			continue
		xmax,ymax = xmin+w, ymin+h
		image_dict[filename]['bbox'].append([xmin,ymin,xmax,ymax,track_id])
		
		if not 'keypoints' in anno:
			continue
		keypoints = anno['keypoints']
		image_dict[filename]['keypoint'].append(keypoints)
		
	#print(image_dict)
	return image_dict

match_list=[13,12,14,9,8,10,7,11,6,3,2,4,1,5,0]
args = parseArgs()

gt_json_path = '/export/home/zby/SiamFC/data/posetrack/posetrack_val.json'
with open(gt_json_path,'r') as f:
	gt_dict = json.load(f)
	
json_dir = '/export/home/zby/SiamFC/data/posetrack/output_json.v3_0_0.2'

image_dir = '/export/home/zby/SiamFC/data/posetrack'
json_files = os.listdir(json_dir)
pose_vis_thresh = args.pose_thresh
pose_number_thresh = args.pose_number
pbar = tqdm(range(len(json_files)))
for json_name in json_files:
	video_name = json_name.replace('.json','')
	if video_name != '002364_mpii_test' and video_name !='001022_mpii_test' and video_name !='006537_mpii_test':
			continue
	video_json = {'annolist':[]}
	gt_video = process_gt(video_name, gt_dict)
	with open(os.path.join(json_dir,json_name),'r') as f:
		old_annolist = json.load(f)['annolist']
	pbar.set_description('Visulizing video {}'.format(video_name))
	color_list = [(255,0,0),(128,0,0),(0,255,0),(0,128,0),(0,0,255),(0,0,128)]
	for i,annotation in enumerate(old_annolist):
		color_flag = 0
		frame_name = annotation['image'][0]['name']
		frame_path = os.path.join(image_dir,frame_name)
		frame = cv2.imread(frame_path)
		im_H, im_W, im_C = frame.shape
		if i==0:
			fourcc = cv2.cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC/data/result/{}.mp4'.format(video_name),fourcc,10,(im_W,im_H))
		old_annorect = annotation['annorect']
		for anno in old_annorect:
			old_point_list = anno['annopoints'][0]['point']
			for pose in old_point_list:
				pose_id, pose_x, pose_y, pose_score = pose['id'][0], pose['x'][0], pose['y'][0], pose['score'][0]
				cv2.circle(frame,(int(pose_x),int(pose_y)),10,color_list[color_flag],-1)
				cv2.putText(frame, str(pose_id), (int(pose_x+5),int(pose_y+5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_list[color_flag], 1)
			xmin, xmax, ymin, ymax, score, track_id = anno['x1'][0], anno['x2'][0], anno['y1'][0], anno['y2'][0], anno['score'][0] ,anno['track_id'][0]
			cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), color_list[color_flag], 2)            
			cv2.putText(frame, 'id:'+str(track_id)+'score:'+'{:.3f}'.format(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_list[color_flag], 1)
			color_flag += 1
			if color_flag >5:
				color_flag = 0
		
		# gt_annotations = gt_video[frame_name]
		# gt_bboxes = gt_annotations['bbox']
		# gt_keypoints = gt_annotations['keypoint']
		# for gt_bbox in gt_bboxes:
			# xmin,ymin,xmax,ymax,track_id = gt_bbox
			# cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
			# cv2.putText(frame, 'id:'+str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
		# for gt_keypoint in gt_keypoints:
			# flag = 0
			# for i in range(17):
				# if i==3 or i==4:
					# continue
				# pose_id = match_list[flag]
				# flag += 1
				# pose_info = gt_keypoint[i*3:(i+1)*3]
				# pose_x, pose_y, pose_vis = pose_info
				# if pose_x ==0 and pose_y==0 and pose_vis==0:
					# continue
				# cv2.circle(frame,(int(pose_x),int(pose_y)),10,(0,255,0),-1)
				# cv2.putText(frame, str(pose_id), (int(pose_x+5),int(pose_y+5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
		videoWriter.write(frame)
	pbar.update(1)
pbar.close()