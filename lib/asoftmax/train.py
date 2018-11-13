import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import resnet as res
import Asoftmax
from torch.utils.data import Dataset
from PIL import Image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image", path)
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms = None, loader = default_loader):
        self.image = [os.path.join(img_path, line.strip().split()[0]) for line in open(txt_path)]
        self.label = [int(line.strip().split()[1]) for line in open(txt_path)]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        imageName = self.image[item]
        label = self.label[item]
        image = self.loader(imageName)
        if self.data_transforms is not None:
            try:
                image = self.data_transforms[self.dataset](image)
            except:
                print("Cannot transform image ", imageName)
        return image, label

class Net():
    
    def __init__(self):
        # data augmentation
        self.data_transforms = {
            "trainImages": transforms.Compose([ 
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
            "testImages": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        }

        # load data
        self.data_dir = "/export/home/dyh/workspace/circle_k/for_douyuhao/data"
        self.train_val = ["trainImages", "testImages"]
        #self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in self.train_val}
        self.image_datasets = {x: customData(img_path = self.data_dir + "/" + x, txt_path = x + ".txt", data_transforms = self.data_transforms, dataset = x) for x in self.train_val}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size = 100, shuffle = True, num_workers = 1) for x in self.train_val}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in self.train_val}
        self.class_nums = 461#self.image_datasets['trainImages'].classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train(self, model, loss_function, optimizer, scheduler, test_step = 1000, save_step = 10000, num_epochs = 25):
        print("==" * 50)
        print("Training starts")
        cnt = 0
        for epoch in range(num_epochs):
            trainLoss1 = 0.0
            trainLoss2 = 0.0
            trainLoss3 = 0.0
            trainLoss4 = 0.0
            trainLoss = 0.0
            trainDataIterator = self.dataloaders['trainImages']
            testDataIterator = self.dataloaders['testImages']
            model.train()
            for inputs, labels in trainDataIterator:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                cnt += 1
                #print(inputs.size())
                #model.train()
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    branch1, branch2, branch3, branch4, output1, output2, output3, output4 = model(inputs)
                    loss1 = loss_function(output1, labels) * 0.3
                    loss2 = loss_function(output2, labels) * 0.3
                    loss3 = loss_function(output3, labels) * 0.3
                    loss4 = loss_function(output4, labels)
                    loss = loss1 + loss2 + loss3 + loss4
                    # backward + optimize in training phase
                    loss.backward()
                    scheduler.step()
                    optimizer.step()
                # statistics
                trainLoss1 += loss1.item()
                trainLoss2 += loss2.item()
                trainLoss3 += loss3.item()
                trainLoss4 += loss4.item()
                trainLoss += loss.item()
                if cnt % test_step == 0:
                    testCnt = 0
                    print("--"*30)
                    print("%d iterations, train loss: %.4f" % (cnt, trainLoss / test_step))
                    print("%d iterations, train loss1: %.4f" % (cnt, trainLoss1 / test_step))
                    print("%d iterations, train loss2: %.4f" % (cnt, trainLoss2 / test_step))
                    print("%d iterations, train loss3: %.4f" % (cnt, trainLoss3 / test_step))
                    print("%d iterations, train loss4: %.4f" % (cnt, trainLoss4 / test_step))
                    print("The learning rate is", optimizer.param_groups[-1]['lr'])
                    trainLoss1, trainLoss2, trainLoss3, trainLoss4, trainLoss = [0.0] * 5
                    model.eval()
                    with torch.no_grad():
                        testLoss1, testLoss2, testLoss3, testLoss4, testLoss = [0.0] * 5
                        for testData, testLabel in testDataIterator:
                            testData = testData.to(self.device)
                            testLabel = testLabel.to(self.device)
                            feature1, feature2, feature3, feature4, output1, output2, output3, output4 = model(testData)
                            loss1 = loss_function(output1, testLabel) * 0.3
                            loss2 = loss_function(output2, testLabel) * 0.3
                            loss3 = loss_function(output3, testLabel) * 0.3
                            loss4 = loss_function(output4, testLabel)
                            loss = loss1 + loss2 + loss3 + loss4
                            testCnt += 1
                            testLoss1 += loss1
                            testLoss2 += loss2
                            testLoss3 += loss3
                            testLoss4 += loss4
                            testLoss += loss
                            if testCnt == 400:
                                print("%d iterations, test loss: %.4f" % (cnt, testLoss / 400.0))
                                print("%d iterations, test loss1: %.4f" % (cnt, testLoss1 / 400.0))
                                print("%d iterations, test loss2: %.4f" % (cnt, testLoss2 / 400.0))
                                print("%d iterations, test loss3: %.4f" % (cnt, testLoss3 / 400.0))
                                print("%d iterations, test loss4: %.4f" % (cnt, testLoss4 / 400.0))
                                print("--" * 30)
                                break
                if cnt % save_step == 0:
                    print("**"*30)
                    print("Save model to resnet_Asoftmax_iter_%d.pt" % cnt)
                    torch.save(model.state_dict(), "model/" + "resnet_Asoftmax_iter_%d.pt" % cnt)
                    print("**" * 30)
                model.train()
            print('Training Done...')
            return model
if __name__ == '__main__':
    net = Net()
    #model = models.inception_v3(pretrained = True)
    model = res.resnet50(pretrained = False, num_classes = 461)
    pretrained_model = models.resnet50(pretrained = True)
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model = model.to(net.device)
    loss_function = Asoftmax.AngleLoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    lr_policy = lr_scheduler.StepLR(optimizer, step_size = 40000, gamma = 0.1)

    model = net.train(model, loss_function, optimizer, lr_policy, save_step = 10000, num_epochs = 3)



