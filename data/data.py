#data augmentation, define image preprocessor
from google.colab import drive
import pathlib

drive.mount('/content/drive', force_remount=True)
drive = pathlib.Path('./drive/MyDrive') / 'ML_Project' / 'ML_FP_2022'


import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2
import random
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
import torchvision.datasets as td
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
])

labels_c = train_['creator']+test_['creator']+eval_['creator']
encoder_c = LabelEncoder()
encoder_c.fit(labels_c)

labels_m = train_['material']+test_['material']+eval_['material']
encoder_m = MultiLabelBinarizer()
encoder_m.fit(labels_m)

labels_t = train_['type']+test_['type']+eval_['type']
encoder_t = MultiLabelBinarizer()
encoder_t.fit(labels_t)

class train_dataset():
  def __init__(self, dict_ = train_, encoder = [encoder_c, encoder_m, encoder_t]):
    #load data
    self.transform = preprocess
    self.image = dict_['identifier']
    self.creator = encoder[0].transform(dict_['creator'])
    self.material = encoder[1].transform(dict_['material'])
    self.art_type = encoder[2].transform(dict_['type'])
    
  def __len__(self):
    return len(self.image)

  def __getitem__(self, index, img_path=drive / "data2022" / "multi-task" / 'data' / "train set new"):
    # img augmentation
    img = cv2.imread(os.path.join(img_path / self.image[index]))
    img = self.transform(img)
    
    #label
    creator = self.creator[index]
    material = self.material[index]
    art_type = self.art_type[index]

    sample = {'image' : img, 'creator' : creator, 'material' : material, 'type' : art_type}
    return sample
  
class test_dataset():
  def __init__(self, dict_ = test_, encoder = [encoder_c, encoder_m, encoder_t]):
    #load data
    self.transform = preprocess
    self.image = dict_['identifier']
    self.creator = encoder[0].transform(dict_['creator'])
    self.material = encoder[1].transform(dict_['material'])
    self.art_type = encoder[2].transform(dict_['type'])
    
  def __len__(self):
    return len(self.image)

  def __getitem__(self, index, img_path=drive / "data2022" / "multi-task" / 'data' / "test set"):
    # img augmentation
    img = cv2.imread(os.path.join(img_path / self.image[index]))
    img = self.transform(img)
    
    #label
    creator = self.creator[index]
    material = self.material[index]
    art_type = self.art_type[index]

    sample = {'image' : img, 'creator' : creator, 'material' : material, 'type' : art_type}
    return sample

class val_dataset():
  def __init__(self, dict_ = eval_, encoder = [encoder_c, encoder_m, encoder_t]):
    #load data
    self.transform = preprocess
    self.image = dict_['identifier']
    self.creator = encoder[0].transform(dict_['creator'])
    self.material = encoder[1].transform(dict_['material'])
    self.art_type = encoder[2].transform(dict_['type'])
    
  def __len__(self):
    return len(self.image)

  def __getitem__(self, index, img_path=drive / "data2022" / "multi-task" / 'data' / "eval set"):
    # img augmentation
    img = cv2.imread(os.path.join(img_path / self.image[index]))
    img = self.transform(img)
    
    #label
    creator = self.creator[index]
    material = self.material[index]
    art_type = self.art_type[index]

    sample = {'image' : img, 'creator' : creator, 'material' : material, 'type' : art_type}
    return sample

train = train_dataset()
test = test_dataset()
val = val_dataset()
