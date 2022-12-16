# Artwork Recognition-JHU601.675


Here is the link to our final report colab:
**[Colab](https://colab.research.google.com/drive/1aiCmiGC7pgI-zHdeYGDP-HsO1pCe2c57)**

## Intrdouction
A museum can contain artworks dating back to more than hundred years ago from over thousands of creators. With the large amount of image data on hand, we can build machine learning models that enable us to accomplish a variety of goals. Some examples include finding trends in artworks throughout the history, instant art image recognition, and artwork generation, etc., all of which are our motivations for pursuing this problem. And we majorly focus on the classification problem and furthermore, multitasking classification.

Our Artwork Recognition project focuses on all kinds of museum artwork image classification. It mainly contains two parts: single-task classification and multitasking classification. We want make better classication model than baseline on single-task classification. Then We according to strong information correlation, do the multitasking classification and outperform than KKN method, which is considered as baseline.

## Single-task Image Classification
There are 3 different single-task classification: type, material and creator. Each task uses specific training images and training model is in different ipynb file saved in google drive.

### Type classification
#### Training images 
Images are saved in Directory:Google Drive:ML_FP_2022/data2022/type

Over 900 training images are in the file 'image_total'. 70 test images are in 'new_test'. 15 evaluation images are in 'evaluate_image'.
#### Training Model file: 
Directory:Google Drive:ML_FP_2022/image_classification/ [最终文件]

So just run the [最终文件] file and the models are saved in ML_FP_2022/models/type

### Material classification
#### Training images 
Images are saved in Directory:Google Drive:ML_FP_2022/data2022/material
#### Training Model file: 
Directory:Google Drive:ML_FP_2022/image_classification/ [最终文件]

So just run the [最终文件] file and the models are saved in ML_FP_2022/models/type

### Creator classification
#### Training images 
Images are saved in Directory:Google Drive:ML_FP_2022/data2022/creator

Labels of the images are also saved in this directory
#### Training Model file: 
Directory:Google Drive:ML_FP_2022/image_classification/ [最终文件]

So just run the [最终文件] file and the models are saved in Google Drive:ML_FP_2022/models/type

## Multi-task Image Classification
Similarly, there will be image selection process as well as a model training fil.
### Image Selection process

### classification model training
