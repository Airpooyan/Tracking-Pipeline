import torch
import numpy as np
import pickle
import os
import cv2
import json
import functools
import xml.etree.ElementTree as ET

from multiprocessing import Pool
from tqdm import tqdm
from glob import glob

from .siamfc import config, get_instance_image

def worker(output_dir, annotation_dir):
    with open(annotation_dir,'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    data_dict = dict()
    for anno in annotations:
        img_id = anno['image_id']
        if not 'bbox' in anno:
            continue
        bbox = anno['bbox']#xmin,ymin,w,h
        track_id = anno['track_id']
        if bbox[2] ==0 or bbox[3] ==0:
            continue
        #bbox[2] /= 1.2
        #bbox[3] /= 1.2
        
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        file_name = str(img_id%1000).zfill(6)
        if img_id not in data_dict:
            data_dict[img_id] = [{'file_name':file_name,'bbox':bbox,'track_id':track_id}]
        else:
            data_dict[img_id].append({'file_name':file_name,'bbox':bbox,'track_id':track_id})
        #print(data_dict)
    #print(annotation_dir)
    video_name = annotation_dir.split('/')[-1].replace('.json','')
    #print(video_name)
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {}
    for image_id,anno in data_dict.items():
        filename = anno[0]['file_name']
        image_dir = annotation_dir.replace('annotations','images').replace('.json','')
        image_file = os.path.join(image_dir,filename)+'.jpg'
        #print(image_file)
        img = cv2.imread(image_file)
        H,W,C = img.shape
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        for obj in anno:
            xmin,ymin,xmax,ymax = obj['bbox']
            if xmin >=W or ymin>=H or xmax<=0 or ymax<=0 or xmin==xmax or ymin==ymax:
                continue
            if xmin<0:
                xmin = 0
            if ymin<0:
                ymin = 0
            if xmax>=W:
                xmax = W-1
            if ymax>=H:
                ymax = H-1
            bbox = [xmin,ymin,xmax,ymax]
            trkid = obj['track_id']
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            instance_img, _, _ = get_instance_image(img, bbox,
                    config.exemplar_size, config.instance_size, config.context_amount, img_mean)
            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs

def processing(data_dir, output_dir, num_threads):
    # get all 4417 videos
    annotation_dir = os.path.join(data_dir,'annotations')
    all_annotations = glob(os.path.join(annotation_dir, 'train/*')) + \
                 glob(os.path.join(annotation_dir, 'val/*'))
   # print(all_annotations)
    meta_data = []
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker, output_dir), all_annotations), total=len(all_annotations)):
            meta_data.append(ret)

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    processing('data/posetrack','data/posetrack/processed_data',2)
