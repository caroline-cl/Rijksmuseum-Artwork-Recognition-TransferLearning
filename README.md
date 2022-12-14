# Artwork Recognition-JHU601.675


Here is the link to our final report colab:
**[Colab](https://colab.research.google.com/drive/1aiCmiGC7pgI-zHdeYGDP-HsO1pCe2c57)**

## Intrdouction
A museum can contain artworks dating back to more than hundred years ago from over thousands of creators. With the large amount of image data on hand, we can build machine learning models that enable us to accomplish a variety of goals. Some examples include finding trends in artworks throughout the history, instant art image recognition, and artwork generation, etc., all of which are our motivations for pursuing this problem. And we majorly focus on the classification problem and furthermore, multitasking classification.

Our Artwork Recognition project focuses on all kinds of museum artwork image classification. It mainly contains two parts: single-task classification and multitasking classification. We want make better classication model than baseline on single-task classification. Then We according to strong information correlation, do the multitasking classification and outperform than KKN method, which is considered as baseline.

## Single-task Image Classification
Under Images_Classification folders
ML_label_create.ipynb is a file that is selected for label when running on google drive.
ML-create-image.ipynb is run locally How to filter image files for label-create files
ML_fp_v3.ipynb is the main file of train

You can visualize your result from our colab notebook. The data for image classification can be downloaded from ./datas

## Multi-task Image Classification
Under Images_Classification folders
ML_label_create.ipynb is a file that is selected for label when running on google drive.
ML-create-image.ipynb is run locally How to filter image files for label-create files
ML_fp_v3.ipynb is the main file of train

You can visualize your result from our colab notebook. The data for image classification can be downloaded from ./datas

## Object Detection
The object detection will give the image with the highlighted box as output which will save as the same path as the input images. The following is an example of the input and the output. 

### Usage
Python object_detection.py -input_file your_image_file -box_info

The -input_file ask user to provide the images path.
The -box_info command can provide you with the coordinates of bounding box in images.

Also user can use evaluation.ipynb to evaluate their result. 

## Style Transfer
we train the model from photo to drawing, drawing to photo, painting to drawing, drawing to painting. We apply adversarial losses to mapping functions. We developing our own script to generate translated images. This parts refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. You can also download prertrained model by download.sh in each folder.

### Usage
pip install -r requirements.txt

Python transform.py --img your_image_file --name model_name

--img is your input image path
--name is your model path which under checkpoints
