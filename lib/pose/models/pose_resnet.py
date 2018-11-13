# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import torch.nn.functional as F
#from torch.autograd import Variable


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class AngleLinear(nn.Module):
	def __init__(self, in_features, out_features, m = 4, phiflag=True):
		super(AngleLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features,out_features))
		self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
		self.phiflag = phiflag
		self.m = m
		self.mlambda = [
			lambda x: x**0,
			lambda x: x**1,
			lambda x: 2*x**2-1,
			lambda x: 4*x**3-3*x,
			lambda x: 8*x**4-8*x**2+1,
			lambda x: 16*x**5-20*x**3+5*x
        ]
		
	def forward(self, input):
		x = input   # size=(B,F)    F is feature len
		w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

		ww = w.renorm(2,1,1e-5).mul(1e5)
		xlen = x.pow(2).sum(1).pow(0.5) # size=B
		wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

		cos_theta = x.mm(ww) # size=(B,Classnum)
		cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
		cos_theta = cos_theta.clamp(-1,1)

		if self.phiflag:
			cos_m_theta = self.mlambda[self.m](cos_theta)
			theta = cos_theta.data.acos()
			k = (self.m*theta/3.14159265).floor()
			n_one = k*0.0 - 1
			phi_theta = (n_one**k) * cos_m_theta - 2*k
		else:
			theta = cos_theta.acos()
			phi_theta = myphi(theta,self.m)
			phi_theta = phi_theta.clamp(-1*self.m,1)

		cos_theta = cos_theta * xlen.view(-1,1)
		phi_theta = phi_theta * xlen.view(-1,1)
		output = (cos_theta,phi_theta)
		return output # size=(B,Classnum,2)

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0 
        self.LambdaMin = 5.0 
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		self.feature_flag = True
		self.inplanes = 64
		self.num_classes = 3594
		self.input_height, self.input_width = 384, 288
		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		super(PoseResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# used for deconv layers
		self.deconv_layers = self._make_deconv_layer(
			extra.NUM_DECONV_LAYERS,
			extra.NUM_DECONV_FILTERS,
			extra.NUM_DECONV_KERNELS,
		)

		self.final_layer = nn.Conv2d(
			in_channels=extra.NUM_DECONV_FILTERS[-1],
			out_channels=cfg.MODEL.NUM_JOINTS,
			kernel_size=extra.FINAL_CONV_KERNEL,
			stride=1,
			padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
		)

		# self.embedding_layer = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
											 # nn.BatchNorm2d(512),
											 # nn.ReLU(True),
											 # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
											 # nn.BatchNorm2d(512),
											 # nn.ReLU(True),
											 # nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
											 # #nn.AvgPool2d(kernel_size=(int(self.input_height/32),int(self.input_width/32)))
											 # )
		self.embedding_layer = nn.Sequential(self._make_embedding_layer(block, 64, 2, stride=1),
											 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
											)
		self.fc_feature = nn.Linear(256 * int(self.input_height/32) * int(self.input_width/32), 2048)
		self.classification = nn.Linear(2048, self.num_classes)
		#self.angleLinear = AngleLinear(512, self.num_classes)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)
		
	def _make_embedding_layer(self, block, planes, blocks, stride=1):
		downsample = nn.Sequential(
			nn.Conv2d(2048, planes * block.expansion,
						 kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
		)

		layers = []
		layers.append(block(2048, planes, stride, downsample))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)
	
	def set_bn_fix(self):
		def set_bn_eval(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
			  m.eval()
		self.bn1.apply(set_bn_eval)
		self.layer1.apply(set_bn_eval)
		self.layer2.apply(set_bn_eval)
		self.layer3.apply(set_bn_eval)
		self.layer4.apply(set_bn_eval)
		self.deconv_layers.apply(set_bn_eval)
		self.final_layer.apply(set_bn_eval)	

	def _get_deconv_cfg(self, deconv_kernel, index):
		if deconv_kernel == 4:
			padding = 1
			output_padding = 0
		elif deconv_kernel == 3:
			padding = 1
			output_padding = 1
		elif deconv_kernel == 2:
			padding = 0
			output_padding = 0

		return deconv_kernel, padding, output_padding

	def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
		assert num_layers == len(num_filters), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'
		assert num_layers == len(num_kernels), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'

		layers = []
		for i in range(num_layers):
			kernel, padding, output_padding = \
				self._get_deconv_cfg(num_kernels[i], i)

			planes = num_filters[i]
			layers.append(
				nn.ConvTranspose2d(
					in_channels=self.inplanes,
					out_channels=planes,
					kernel_size=kernel,
					stride=2,
					padding=padding,
					output_padding=output_padding,
					bias=self.deconv_with_bias))
			layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
			layers.append(nn.ReLU(inplace=True))
			self.inplanes = planes

		return nn.Sequential(*layers)
	
	'''
	flag 0: only pose
	flag 1: only embedding feature
	flag 2: pose and embedding,
	flag 3: tracking feature
	'''
	def forward(self, x, flag=0):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		#print(x.shape)

		if flag == 0:
			x = self.deconv_layers(x)
			x = self.final_layer(x)
			return x
		elif flag ==1:
			x = self.embedding_layer(x)
			x = x.view(x.size(0), -1)
			x = self.fc_feature(x)
			if not self.feature_flag:
				x = self.classification(x)
			return x
		elif flag ==2:
			x1 = self.deconv_layers(x)
			x1 = self.final_layer(x1)
			
			x2 = self.embedding_layer(x)
			x2 = x2.view(x2.size(0),-1)
			x2 = self.fc_feature(x2)
			if not self.feature_flag:
				x2 = self.classification(x2)
			return x1, x2
		elif flag ==3:
			return x
		

	def init_weights(self, pretrained=''):
		if os.path.isfile(pretrained):
			logger.info('=> init deconv weights from normal distribution')
			for name, m in self.deconv_layers.named_modules():
				if isinstance(m, nn.ConvTranspose2d):
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					if self.deconv_with_bias:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			logger.info('=> init embedding weights from normal distribution')
			
			for m in self.final_layer.modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					nn.init.constant_(m.bias, 0)
			
			for name, m in self.embedding_layer.named_modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					#nn.init.constant_(m.bias, 0)
						
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			logger.info('=> init final layer weights from normal distribution')
			


			pretrained_state_dict = torch.load(pretrained)
			logger.info('=> loading pretrained model {}'.format(pretrained))
			self.load_state_dict(pretrained_state_dict, strict=False)
		else:
			logger.error('=> imagenet pretrained model dose not exist')
			logger.error('=> please download it first')
			raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, flag=2, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
