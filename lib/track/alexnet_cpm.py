import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from .config import config
# from pose.models.pose_resnet import get_pose_net
from .cpm import CPM


class SiameseAlexNet(nn.Module):
    def __init__(self, gpu_id, train=True):
        super(SiameseAlexNet, self).__init__()
        # self.PoseNet = get_pose_net(cfg, is_train=False, flag = 4) # Change to the CPM model
        self.PoseNet = CPM(1)
        self.exemplar_size = (8,8)
        self.instance_size = (24,24)
        self.multi_instance_size = [(24,24),(22,22)]
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
            
        self.conv_pose = nn.Conv2d(2048, 384, 3, 1, padding=1)
        self.conv_final = nn.Conv2d(384, 256, 3, 1, groups=2)
        self.conv_cpm_pose = nn.Conv2d(128, 384, 3, 1, padding=1)
        
        self.corr_bias = nn.Parameter(torch.zeros(1))
        if train:
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz),mode='train')
            with torch.cuda.device(gpu_id):
                self.train_gt = torch.from_numpy(gt).cuda()
                self.train_weight = torch.from_numpy(weight).cuda()
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz), mode='valid')
            with torch.cuda.device(gpu_id):
                self.valid_gt = torch.from_numpy(gt).cuda()
                self.valid_weight = torch.from_numpy(weight).cuda()
        self.exemplar = None

    """
    Initial the models 
    """
    def init_models(self, pose_model_file = '/export/home/cyh/modify/openpose_model/cpm.pth', track_model_file = '/export/home/cyh/modify/SiamFC/models/siamfc_pretrained.pth'):

        # Loading the Pose model
        pose_model_file = cfg.TEST.MODEL_FILE if pose_model_file is None else pose_model_file
        print("OpenPose: Loading checkpoint from %s" % (pose_model_file))
        checkpoint=torch.load(pose_model_file)
        model_dict = self.PoseNet.state_dict()
        new_state_dict = OrderedDict()
        # k,v represents the Key,value
        for k,v in checkpoint.items():
            new_name = k[7:] if 'module' in k else k
            if new_name in model_dict:
                new_state_dict[new_name]=v

        model_dict.update(new_state_dict)
        self.PoseNet.load_state_dict(model_dict)
        print('OpenPose: OpenPose network has been initilized') 
        print('-----------------------------------------------------------')
        
        # Loading the Pretrained track model
        print("Tracking: Loading checkpoint from %s" % (pose_model_file))
        checkpoint=torch.load(track_model_file)
        model_dict = self.features.state_dict()
        new_state_dict = OrderedDict()
        for k,v in checkpoint.items():
            new_name = k[9:] if 'feature' in k else k
            if new_name in model_dict:
                new_state_dict[new_name]=v
        model_dict.update(new_state_dict)
        self.features.load_state_dict(model_dict)
        print('Tracking: Tracking network has been initilized') 
        print('-----------------------------------------------------------')
        
        # Fix the Pose Net
        for p in self.PoseNet.parameters():
            p.requires_grad = False
            
    def set_bn_fix(self):
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
              m.eval()
        self.PoseNet.apply(set_bn_eval)
        
    def forward(self, x =(None, None), y = (None,None), feature = None):
        exemplar, instance = x # Tracking input
        exemplar_pose, instance_pose = y # Pose input
        if feature is None:
            if exemplar is not None and instance is not None:
                batch_size = exemplar.shape[0]
                
                exemplar_pose_feature = self.conv_cpm_pose(self.PoseNet(exemplar_pose))
                instance_pose_feature = self.conv_cpm_pose(self.PoseNet(instance_pose))
                #print('the exampeler pose feature is {}, the instance pose feature is {}'.format(exemplar_pose_feature.shape, instance_pose_feature.shape))

                exemplar_pose_feature = F.upsample(exemplar_pose_feature, size= self.exemplar_size, mode='bilinear')
                instance_pose_feature = F.upsample(instance_pose_feature, size= self.instance_size, mode='bilinear')
                #print(exemplar_pose_feature.shape, instance_pose_feature.shape)
                temp_exemplar = self.features(exemplar)
                temp_instance = self.features(instance)
                #print('The exemplar featrue is {}, the instance feature is {}'.format(temp_exemplar.shape, temp_instance.shape))
                instance = self.conv_final(temp_instance + instance_pose_feature)
                exemplar = self.conv_final(temp_exemplar + exemplar_pose_feature)
                
                # exemplar = self.conv_final(self.features(exemplar) + self.conv_pose(exemplar_pose_feature))
                # instance = self.conv_final(self.features(instance) + self.conv_pose(instance_pose_feature))
                
                score_map = []
                N, C, H, W = instance.shape
                #print(instance.shape)
                if N > 1:
                    for i in range(N):
                        score = F.conv2d(instance[i:i+1], exemplar[i:i+1]) * config.response_scale + self.corr_bias
                        score_map.append(score)
                    return torch.cat(score_map, dim=0)
                else:
                    return F.conv2d(instance, exemplar) * config.response_scale + self.bias
            elif exemplar is not None and instance is None:
                exemplar_pose_feature = self.conv_cpm_pose(self.PoseNet(exemplar_pose))
                exemplar_pose_feature = F.upsample(exemplar_pose_feature, size= self.exemplar_size, mode='bilinear')
                exemplar = self.conv_final(self.features(exemplar) + exemplar_pose_feature)
                # inference used
                #self.exemplar = self.features(exemplar)
                return exemplar
            else:
                # inference used we don't need to scale the reponse or add bias
                instance_pose_feature = self.conv_cpm_pose(self.PoseNet(instance_pose))
                instance_pose_feature = F.upsample(instance_pose_feature, size= self.instance_size, mode='bilinear')
                instance = self.conv_final(self.features(instance) + instance_pose_feature)
                score_map = []
                for i in range(instance.shape[0]):
                    score_map.append(F.conv2d(instance[i:i+1], self.exemplar))
                return torch.cat(score_map, dim=0)
        else:
            self.exemplar = feature
            N, C, H, W = instance.shape
            #print(H,W)
            if H == 255:
                instance_flag = 0
            elif H == 239:
                instance_flag = 1
            instance_pose_feature = self.conv_cpm_pose(self.PoseNet(instance_pose))
            self.instance_size
            instance_pose_feature = F.upsample(instance_pose_feature, size= self.multi_instance_size[instance_flag], mode='bilinear')
            #print(instance_pose_feature.shape)
            instance = self.conv_final(self.features(instance) + instance_pose_feature)
            score_map = []
            for i in range(instance.shape[0]):
                score_map.append(F.conv2d(instance[i:i+1], self.exemplar))
            return torch.cat(score_map, dim=0)      

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            #print(pred.shape,self.train_gt.shape)
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                    self.train_weight, size_average = False) / config.train_batch_size # normalize the batch_size
        else:
            #print(pred.shape, self.valid_gt.shape, self.valid_weight.shape)
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                    self.valid_weight, size_average = False) / config.valid_batch_size # normalize the batch_size

    def _create_gt_mask(self, shape, mode='train'):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        if mode == 'train':
            mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        elif mode == 'valid':
            mask = np.repeat(mask, config.valid_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)
