###main train and evaluate function ###
from tqdm import tqdm
import time
import os
import pickle
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import cv2
import random
import torchvision.transforms as transforms
from torch.utils import data
from torch.nn import functional as F
from torchvision import models
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import numpy as np
import torch.optim as optim
import torchvision.datasets as td
from torchvision import transforms
from PIL import Image
from skimage.feature import hog as hog
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix,f1_score, accuracy_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder


def train_main(model,loss_,optimizer,train_loader,device, epoch):
    loss_epoch=[]
    loss_each=[[],[],[]]
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
      for data in tepoch:
        ##(b,c,h,w)
        tepoch.set_description(f"Epoch {epoch}")
        images = torch.Tensor(data['image']).to(device)
        label_c = torch.Tensor(data['creator']).to(device)
        label_m = torch.Tensor(data['material']).to(device)
        label_t = torch.Tensor(data['type']).to(device)
        #print(images.shape)
        optimizer.zero_grad()
        outputs_c, outputs_m, outputs_t = model(images)
        outputs_m=outputs_m.unsqueeze(1).float()
        outputs_t=outputs_t.unsqueeze(1).float()
        label_m=label_m.unsqueeze(1).float()
        label_t=label_t.unsqueeze(1).float()

        loss_c = loss_[0](outputs_c, label_c)
        loss_m = loss_[1](outputs_m, label_m)
        loss_t = loss_[2](outputs_t, label_t)
        print(loss_c.detach().cpu().numpy().ravel(), loss_m.detach().cpu().numpy().ravel() , loss_t.detach().cpu().numpy().ravel())
        loss_each[0].append(loss_c.item())
        loss_each[1].append(loss_m.item())
        loss_each[2].append(loss_t.item())
        # total_loss
        loss= loss_c + 0.08*loss_m + loss_t
        #loss = loss_[1](outputs_m, label_m) + loss_[2](outputs_t, label_t)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()

        tepoch.set_postfix(loss=loss.item())
        time.sleep(0.0001)

  return loss_epoch,loss_each

def evaluate_whole(model,evaluate_loader):
    predict_clist=[]
    predict_mlist=[]
    predict_tlist=[]
    y_c = []
    y_m = []
    y_t = []
    model.eval()
    with torch.no_grad():
      for batchind,data in enumerate(evaluate_loader):
        images = torch.Tensor(data['image']).to(device)
        label_c = torch.Tensor(data['creator']).to(device)
        label_m = torch.Tensor(data['material']).to(device)
        label_t = torch.Tensor(data['type']).to(device)

        outputs_c, outputs_m, outputs_t = model(images)
        softmax = nn.Softmax()
        outputs_c = softmax(outputs_c).argmax(dim=1, keepdim=True)
        outputs_m=outputs_m.float()
        outputs_t=outputs_t.float()
        label_c=label_c
        label_m=label_m.float()
        label_t=label_t.float()
        ##change to (1,c,h,w)

        # predict creator
        predict_clist.append(outputs_c.detach().cpu().numpy().ravel())
        y_c.append(torch.Tensor.round(label_c).detach().cpu().numpy().ravel())
        # predict material
        predict_mlist.append(torch.Tensor.round(outputs_m).detach().cpu().numpy().ravel())
        y_m.append(torch.Tensor.round(label_m).detach().cpu().numpy().ravel())
        # predict type
        predict_tlist.append(torch.Tensor.round(outputs_t).detach().cpu().numpy().ravel())
        y_t.append(torch.Tensor.round(label_t).detach().cpu().numpy().ravel())
    print(y_m[0].shape)
    print(predict_mlist[0].shape)
    print(y_t[0].shape)
    print(predict_tlist[0].shape) 

    #Artist aacc
    aacc = 0
    for index in range(len(predict_clist)):
      aacc = aacc + accuracy_score(y_c[index], predict_clist[index])
    aacc = aacc / len(predict_clist)

    #material mmAP
    mmAp = 0
    for index in range(len(predict_mlist)):
      mmAp = mmAp + average_precision_score(y_m[index], predict_mlist[index])
    mmAp = mmAp / len(predict_mlist)

    #type tmAP
    tmAp = 0
    for index in range(len(predict_tlist)):
      tmAp = tmAp + average_precision_score(y_t[index], predict_tlist[index])
    tmAp = tmAp / len(predict_tlist)

    return [predict_clist, predict_mlist, predict_tlist], [y_c, y_m, y_t], [aacc, mmAp, tmAp]
