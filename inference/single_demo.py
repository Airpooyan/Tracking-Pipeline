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
from tracker import SiamFCTracker

json_file_path = '/export/home/zby/SiamFC-PyTorch/data/posetrack/annotations/val' 

def main(video_dir, gpu_id =0,  model_path='/export/home/zby/SiamFC-PyTorch/models/siamfc_pretrained.pth'):
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
    #    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2]
        if bbox[2]==0 or bbox[3]==0:
            continue
        #bbox[2] /= 1.2
       # bbox[3] /= 1.2
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        #bbox = [bbox[0]-bbox[2]/2, bbox[1]-bbox[3]/2, bbox[2], bbox[3]]
        if not track_id in bbox_dict:
            bbox_dict[track_id] = {'bbox':bbox,'frame_id':frame_id} 
    image_path = json_file.replace('annotations','images').replace('.json','')
    filenames = sorted(glob.glob(os.path.join(image_path, "*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    #print(filenames)
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    print('Images has been loaded')
    im_H,im_W,im_C = frames[0].shape
    videoWriter = cv2.VideoWriter('/export/home/zby/SiamFC-PyTorch/data/KiteSurf/result/{}.avi'.format(json_name.replace('.json','')),cv2.cv2.VideoWriter_fourcc('M','J','P','G'),10,(im_W,im_H))

    detector = Detector()
    title = json_name.replace('.json','')
    # starting tracking
    print(bbox_dict)
    tracker = SiamFCTracker(model_path, gpu_id)
    predict_dict = dict()
    for track_id, anno in bbox_dict.items():
        start_box = anno['bbox']
        start_frame = anno['frame_id']
        frame = frames[start_frame]
        #print(frame)
        tracker.init(frame,start_box)
        print('First frame has been initinized')
        for idx, frame in enumerate(frames):
            bbox,score = tracker.update(frame)
            #print(score)
            #print(bbox)
            if not idx in predict_dict:
                predict_dict[idx] = []
            bbox.append(track_id)
            bbox.append(score)
            predict_dict[idx].append(bbox)
            #print(predict_dict)
        # bbox xmin ymin xmax ymax
    score_thresh = 4e-5
    for frame_id,bboxs in predict_dict.items():
        #print(frame_id)
        frame = frames[frame_id]
        #print(bboxs)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for bbox in bboxs:
            xmin,ymin,xmax,ymax,idx,score = bbox
            if score< score_thresh:
                cv2.rectangle(frame,
                              (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)),
                              (0, 0, 255), 
                              2)
            else:                  
                cv2.rectangle(frame,
                              (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)),
                              (0, 255, 0),
                              2)            
            if score < score_thresh:
                cv2.putText(frame, 'id:'+str(idx)+' score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            else:
                cv2.putText(frame, 'id:'+str(idx)+' score:'+str(score), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        #cv2.imwrite('/export/home/zby/SiamFC-PyTorch/data/KiteSurf/result/{}_{}.jpg'.format(json_name.replace('.json',''),frame_id), frame)
        videoWriter.write(frame)
        if frame_id==0:
            for idx,anno in bbox_dict.items():
                xmin,ymin,w,h = anno['bbox']
                xmax,ymax = xmin+w,ymin+h
                cv2.rectangle(frame,
                              (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)),
                              (255, 0, 0), 
                              2)
                cv2.imwrite('/export/home/zby/SiamFC-PyTorch/data/KiteSurf/result/{}_{}.jpg'.format(json_name.replace('.json',''),frame_id), frame)


if __name__ == "__main__":
    main('/export/home/zby/SiamFC-PyTorch/data/KiteSurf',model_path='/export/home/zby/SiamFC/models/output/siamfc_35.pth')
