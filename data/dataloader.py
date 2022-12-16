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

BATCH_SIZE = 16
BATCH_SIZE_VAL = 1

train_dataloader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test, shuffle=False, batch_size=BATCH_SIZE_VAL)
val_dataloader = DataLoader(val, shuffle=False, batch_size=BATCH_SIZE_VAL)
