import os
import numpy as np
import torch
import torchvision
import resnet as res
import time
from PIL import Image
from load import dataLoader, dataAugmentation
class modelEval():
    def __init__(self, model, state_dict):
        self.model = model
        self.weight = state_dict
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def loadWeight(self):
        self.model.load_state_dict(torch.load(self.weight))
        self.model.eval()
        model = self.model.to(self.device)
        return model
    def loadData(self, dataPath, dataList, dataset = ''):
        augmentation = dataAugmentation()
        data = dataLoader(dataPath, dataList, dataset = dataset, data_transforms = augmentation.data_transforms)
        return data
    def singleSearchL2(self):
        rightCnt = 0
        t0 = time.time()
        for i in range(len(self.test)):
            label = self.testLabel[i]
            feature = self.test[i]
            dist = np.sum(np.square(self.lib - feature), axis = 1)
            index = dist.argsort(axis = 0)[0]
            top1 = self.libLabel[int(index)]
            if top1 == label:
                rightCnt += 1
        t1 = time.time()
        print("Average time is %f" % ((t1 - t0) / len(self.test)))
        print("Precision is %f" % (rightCnt / len(self.test)))
        
    def getMatrix(self, path, listName, fileName):
        data = self.loadData(path, listName, 'testImages')
        dataloaders = torch.utils.data.DataLoader(data, batch_size = 1)
        t = 0.0
        matrix = []
        labels = []
        cnt = 0
        for image, label in dataloaders:
            cnt += 1
            with torch.no_grad():
                image = image.to(self.device)
                t0 = time.time()
                feature1, feature2, feature3, feature4, output1, output2, output3, output4 = model(image)
                t1 = time.time()
                t += t1 - t0
                features = feature4.to("cpu")
                f = features.numpy()
                matrix.append(f[0])
                labels.append(label)
                if cnt % 10000 == 0:
                    print("%d images" % cnt)
        print("All %d images" % cnt)
        matrix = np.stack(matrix, axis = 0)
        labels = np.array(labels)
        print(labels)
        print(labels.shape)
        np.savetxt(fileName, matrix, fmt = "%f", delimiter = " ")
        print("There are %d images. Average time of each iteration is %f" % (cnt, t / cnt))
        return matrix, labels
    def evaluate(self, searchPath, searchList, testPath, testList):
        self.model = self.loadWeight()
        print("Start building search pool")
        self.lib, self.libLabel = self.getMatrix(searchPath, searchList, "dataset.txt")
        print("Build search pool successfully.")
        print("Start test images")
        self.test, self.testLabel = self.getMatrix(testPath, testList, "testset.txt")
        print("All features of test images are saved.")
        self.singleSearchL2()

if __name__ == "__main__":
    model = res.resnet50(pretrained = False, num_classes = 461)
    instance = modelEval(model, "model/resnet_Asoftmax_iter_100000.pt")
    instance.evaluate("/export/home/dyh/workspace/circle_k/for_douyuhao/data/searchImages/", "searchListPart.txt", "/export/home/dyh/workspace/circle_k/for_douyuhao/data/testImages/", "testList.txt")

