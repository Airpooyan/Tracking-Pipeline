import json
import os
import argparse
from tqdm import tqdm

def parseArgs():

    parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
    parser.add_argument("-t", "--thresh",dest = 'det_thresh',required=False, default=0.6, type=float)
    parser.add_argument("-d", "--testdir",dest = 'test_dir',required=False, default=None, type=str)
    return parser.parse_args()

txt = open('result.txt','a')
args = parseArgs()

if args.test_dir == None:
	test_filedir = 'evaluate_pose_tf_dt'
else:
	test_filedir = args.test_dir

json_dir = '/export/home/zby/SiamFC/data/posetrack/'+test_filedir
json_files = os.listdir(json_dir)
det_thresh = args.det_thresh
print('--------------------------------------')
print('DetThresh: {}'.format(det_thresh))

txt.write('--------------------------------------\n')
txt.write('TestMethod: {} DetThresh\n'.format(test_filedir, det_thresh))

pbar = tqdm(range(len(json_files)))
save_dir = '/export/home/zby/SiamFC/data/posetrack/{}_detThresh{}'.format(test_filedir, det_thresh)
if not os.path.exists(save_dir):
	os.mkdir(save_dir)
for json_name in json_files:
	video_json = {'annolist':[]}
	with open(os.path.join(json_dir,json_name),'r') as f:
		old_annolist = json.load(f)['annolist']
	save_path = os.path.join(save_dir, json_name)
	pbar.set_description('Processing video {}'.format(json_name))
	pbar.update(1)
	for annotation in old_annolist:
		#print(frame_name)
		image_dict = dict()
		old_annorect = annotation['annorect']
		new_annorect = []
		for anno in old_annorect:
			det_score = anno['score'][0]
			if det_score >= det_thresh:
				new_annorect.append({'x1':anno['x1'],'x2':anno['x2'],'y1':anno['y1'],'y2':anno['y2'],'score':anno['score'],'track_id':anno['track_id'],'annopoints':anno['annopoints']})
		image_dict['image'] = annotation['image']
		image_dict['annorect'] = new_annorect
		video_json['annolist'].append(image_dict)
	with open(save_path,'w') as f:
		json.dump(video_json, f)
pbar.close()
txt.close()
print('result has been saved to {}'.format(save_dir))