import numpy as np
import torchvision
from torchvision import transforms as T
import torchmetrics as tm
import os
import cv2
import json
import shutil
import torch

import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

from scipy.ndimage import zoom
from scipy.stats import rayleigh
from tqdm import tqdm

import ultralytics
from ultralytics import YOLO

from mapcalc import calculate_map, calculate_map_range

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image

######################### Image Manipulation Functions ##################

# function #1 to perform best-practice enhancements
#   expects img to be loaded as cv2 / numpy array (h x w x 3)
#
def rayleigh_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    # Convert to float for processing
    
    img = img.astype(np.float64)
    
    # Normalize the image to range [0,1]
    img /= np.max(img)
    
    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply((img*255).astype(np.uint8))
    
    # Calculate scale for Rayleigh distribution based on the image
    scale = np.std(img_clahe)
    
    # Flatten the image for processing
    img_flat = img_clahe.flatten()
    
    # Generate Rayleigh distributed histogram
    rayleigh_hist, bin_edges = np.histogram(rayleigh.rvs(scale=scale, size=len(img_flat)), bins=256)
    
    # Generate the histogram of the input image
    img_hist, _ = np.histogram(img_flat, bins=256)
    
    # Calculate the CDFs
    img_cdf = img_hist.cumsum() / img_hist.sum()
    rayleigh_cdf = rayleigh_hist.cumsum() / rayleigh_hist.sum()
    
    # Perform histogram matching
    img_matched = np.interp(img_flat, bin_edges[:-1], rayleigh_cdf)
    
    # Map the matched image to the CDF of the original image
    img_matched = np.interp(img_matched, img_cdf, bin_edges[:-1])
    
    # Reshape the image back to its original shape
    img_matched = img_matched.reshape(img.shape)
    
    return img_matched

# function #2 to perform best-practice enhancements
def lanczos_interpolation(img, zoom_factor):
    # Perform Lanczos-3 interpolation
    img_interpolated = zoom(img, zoom_factor, order=3)
    
    return img_interpolated

# function to take a single images and run cleanup
#   accepts a path to the image and returns a cleaned image
#   expected options:
#   - CLAHE + noise reduction by interpolation
#   - resize to standard or custom
#   - trim borders which don't have data
#
def cleanup_image(imgpath, path = False, resize = False, targetsize = (224,224)
                  ,crop = False, crop_xyxy = (0,0,224,224), verbose = False):

    # optionally pass the path or an image object itself
    if path:
        img = cv2.imread(imgpath)
    else:
        img = imgpath
    if verbose:
        print("Dimensions of image:",img.shape)

    if crop:
        img = img[crop_xyxy[0]:crop_xyxy[2],crop_xyxy[1]:crop_xyxy[3],:]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE contrast enhancement
    assert img is not None, "file could not be read, check whether os.path.exists()"
    img_eq = rayleigh_clahe(img)
    img_int = lanczos_interpolation(img_eq, 1)  # No Zoom in

    # resize
    if resize:
        img_cfr = cv2.resize(img_int, targetsize, interpolation=cv2.INTER_CUBIC)
        if verbose:
            print("Dimensions after resize:",img_cfr.shape)
            print("Type after resize:",img_cfr.dtype)
    else:
        img_cfr = img_int

    # trim_margins: optional parameter, 
    #   if we wanted to add in the function to trim the ultrasound annotation stuff here

    # optionally show the progression of images
    if verbose:
        images = cv2.hconcat([img,img_int])
        #plt.imshow(img)
        plt.imshow(images,cmap='gray')
        #plt.imshow(img_cfr)
        plt.show()
        plt.imshow(img_cfr,cmap='gray')
        plt.show()

    img_cfr = cv2.cvtColor(img_cfr, cv2.COLOR_GRAY2RGB)

    return img_cfr

# DEPRECATED - NOT USED
# function to enhance input images
#   accepts a list of image paths, train/test split, and destination folder
#   enhances each image and saves them
#
def cleanup_imlist(img_split_list, output_folder):

    for img_path in tqdm(img_split_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_eq = cleanup_image(img_path)

        if img_path.split("\\")[-2] == "train":
            # Save the image to train folder
            cv2.imwrite(os.path.join(output_folder, 'train/' + os.path.basename(img_path)), img_eq)
        elif img_path.split("\\")[-2] == "val":
            # Save the image to val folder
            cv2.imwrite(os.path.join(output_folder, 'val/' + os.path.basename(img_path)), img_eq)

# function to enhance selected images and save to disk
#   accepts a dataframe with ['image_path'] and ['image'] as columns
#   and an output folder destination
#   if output_folder exists already, an incrementally named new one will be created
#
def enhance_images(img_df, output_folder):

    img_enh_df = img_df.copy()
    i = ''

    # check if folder exists, increment as needed
    if os.path.exists(output_folder):
        i = 1
        while os.path.exists(output_folder+str(i)):
            i+=1

    output_folder = output_folder+str(i)+'\\'
    os.makedirs(output_folder)
    print('Created folder for output images:')
    print(output_folder)

    # loop through all images
    for i in range(img_df.shape[0]):
        img_enh = cleanup_image(img_df['image_path'][i])
    
        # Save the enhanced image to output folder
        img_enh_path = output_folder + img_df['image_id'][i] + '.jpg'
        cv2.imwrite(img_enh_path, img_enh)
        img_enh_df['image_path'][i] = img_enh_path

        # Save the corresponding json to same output folder
        json_path = img_df['image_path'][i].replace('.jpg','.json').replace('jpgs','jsons')
        with open(json_path, 'r') as f:
            data = json.load(f)
        new_json_path = output_folder + img_df['image_id'][i] + '.json'
        with open(new_json_path, 'w') as f:
            json.dump(data, f)

    # return the updated dataframe with new paths
    return img_enh_df

######################### Metrics Functions ################################

# function to compare two bounding boxes and return DSC score
#   DSC = Dice Similarity Coefficient
#   inputs should be dictionaries or pandas dfs
#   with keys x1, x2, y1, y2 for top left and bottom right points
#   output is a score between 0~1 as a float
#   adapted from this StackOverflow post:
#   https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
#
def dsc_score(box1, box2):

    assert box1['x1'] < box1['x2']
    assert box1['y1'] < box1['y2']
    assert box2['x1'] < box2['x2']
    assert box2['y1'] < box2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(box1['x1'], box2['x1'])
    y_top = max(box1['y1'], box2['y1'])
    x_right = min(box1['x2'], box2['x2'])
    y_bottom = min(box1['y2'], box2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    # compute the DSC score = 2 * intersection / (box1 + box2)
    dsc = 2.0 * intersection_area / float(box1_area + box2_area)
    assert dsc >= 0.0
    assert dsc <= 1.0
    return dsc

# function to calculate DSC scores of two dataframes of segmentation labels
#   labels should be bounding boxes with 'x1','x2','y1','y2' elements
#
def score_labels_dsc(label_df,pred_df,verbose=False):
    
    # initialize variables
    scores = []

    # confirm lists are compatible
    ll = len(label_df)
    lp = len(pred_df)

    if ll == lp:

        # collect DSC scores for each pair
        for i in range(ll):
            scores.append(dsc_score(label_df.iloc[i],pred_df.iloc[i]))
            
            # optionally display image with annotation boxes
            if verbose > 1:
                display_img_ann(label_df.iloc[i],pred_df.iloc[i])

    else:
        l1 = list(label_df['image_id'])
        l2 = list(pred_df['image_id'])
        print("Missing images:", (set(l1).difference(l2)))
        print("Missing labels:", (set(l2).difference(l1)))

    return scores

def display_img_ann(label_df_row, pred_df_row):

    # Load the image
    img = cv2.imread(label_df_row['image_path'])
    print('Image name:',label_df_row['image_id'])
    #print(label_df_row['image_path'])

    # Load the box corners for the true labels
    lx1 = int(label_df_row['x1'])
    ly1 = int(label_df_row['y1'])
    lx2 = int(label_df_row['x2'])
    ly2 = int(label_df_row['y2'])
    # Load the box corners for the predicted labels
    px1 = int(pred_df_row['x1'])
    py1 = int(pred_df_row['y1'])
    px2 = int(pred_df_row['x2'])
    py2 = int(pred_df_row['y2'])

    print('ground truth bbox:',lx1,ly1,lx2,ly2)
    print('labeled bbox:',px1,py1,px2,py2)

    # Box color and label depend on label
    if label_df_row['label_bool'] == 0:
        ltumor = "l-benign"
        lcolor = (0, 255, 0)
    elif label_df_row['label_bool'] == 1:
        ltumor = "l-malignant"
        lcolor = (0, 0, 255)

    # Box color and label depend on label
    if pred_df_row['label_bool'] == 0:
        ptumor = "p-benign"
        pcolor = (0, 255, 0)
    elif pred_df_row['label_bool'] == 1:
        ptumor = "p-malignant"
        pcolor = (0, 0, 255)

    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), lcolor, 2)
    cv2.rectangle(img, (px1, py1), (px2, py2), pcolor, 2)

    # Prepare the labels
    ltext = f"{ltumor}"
    ptext = f"{ptumor}"

    # Draw the label and confidence
    cv2.putText(img, ltext, (lx1, ly1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, lcolor, 2, cv2.LINE_AA)
    cv2.putText(img, ptext, (px1, py2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, pcolor, 2, cv2.LINE_AA)

    cv2.imwrite('temp.jpg',img)
    #cv2.imshow(label_df_row['image_id'],img)
    # show annotated image
    #plt.imshow(img, cv2.COLOR_BGR2RGB)
    #plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis

    img = Image.open('temp.jpg').convert('RGB')
    plt.imshow(img)
    plt.show()

#### TODO figure out how to show the image!!

    return

# function to calculate mean Average Precision scores
#   for dataframes of true labels and predicted labels
#   using mapcalc library
#
def map_scores(ground_truth, result_dict):

    map50 = calculate_map(ground_truth, result_dict, 0.5)
    map5090 = calculate_map_range(ground_truth, result_dict, 0.5, 0.95, 0.05)

    return [map50, map5090]


################################ IO Functions ################################

# function to load the AI+ provided bounding boxes and diagnosis labels
#   accepts a folder path containing the .json files in subdirectories
#   returns a dataframe of bounding points and a list of unique image IDs
#
def load_aiplus_labels(folder_path):
    # Initialize variables
    to_df = []

    # Collect all JSON file paths in the target folder
    path_list = glob(folder_path+'\\**\\*.json',recursive=True)

    # if no JSONs found, assume there are yoloformat text label files
    if len(path_list) == 0:
        # Collect all text file paths in the target folder
        txt_list = glob(folder_path+'\\**\\*.txt',recursive=True)

        for path in txt_list:
            with open(path,'r') as f:
                data = f.read().split(' ')
                
            # get the diagnosis label from the json
            diag_label = 'benign' if data[0] == 0 else 'malignant'

            # get the bounding box points
            pts = [float(pt) for pt in data[1:]]

            # get the image base name without extension
            imname = path.split('\\')[-1].replace('.txt','')

            # get the image path (not the label path)
            impath = path.replace('.txt','.jpg').replace('labels','images')
            if not os.path.exists(impath):
                impath = impath.replace('.jpg','.png')

            # load the image and get its shape
            h = 0
            w = 0
            img = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            # convert the % labels to pixel locations
            x1 = int(pts[0] * w)
            y1 = int(pts[1] * h)
            x2 = int(pts[2] * w)
            y2 = int(pts[3] * h)

            # append to dataframe
            to_df.append([imname,impath,x1,y1,x2,y2,diag_label])
            f.close()

    # otherwise process the JSONs
    else:
        # Loop through each file
        for path in path_list:

            # load the json file
            with open(path,'r') as json_file:
                data = json.load(json_file)

            # get the diagnosis label from the json
            #   or from the parent folder as a backup
            if data['shapes'][0]['label'] == 'tumor':
                diag_label = path.split('\\')[-3]
            else:
                diag_label = data['shapes'][0]['label']

            # get the image base name without extension
            imname = path.split('\\')[-1].replace('.json','')
            
            # get the bounding box points
            pts = data['shapes'][0]['points']
            
            # get the image path (not the label path)
            impath = path.replace('.json','.jpg').replace('jsons','imgs')

            # append to dataframe
            to_df.append([imname,impath,int(pts[0][0]),int(pts[0][1]),int(pts[1][0]),int(pts[1][1]),diag_label])
            json_file.close()

    df = pd.DataFrame(to_df,columns=['image_id','image_path','x1','y1','x2','y2','label'])
    labeled_imlist = df['image_id'].unique()

    # convert label from text to boolean
    df['label_bool'] = [0 if df.loc[i,'label'] == 'benign' else 1 for i in range(len(df))]
    # df = df.drop(['label'],axis=1)

    return df, labeled_imlist

# function to load the paths of AI+ provided images
#   accepts a folder path containing the .jpg files in subdirectories
#   returns a dataframe of images + paths and a list of unique image IDs
#   optionally accepts list of image IDs (for example, those with provided labels):
#       if len(list)>0, will only return image paths with matching IDs
#
def load_aiplus_images(folder_path,labeled_imlist=[]):
    # Initialize variables
    to_df = []

    # Collect all JSON file paths in the target folder
    path_list = glob(folder_path+'\\**\\*.jpg',recursive=True)

    # Loop through each file
    for path in path_list:
       
        # get the image base name without extension
        imname = path.split('\\')[-1].replace('.jpg','')

        # if labels are provided, only evaluate images that have a label
        if len(labeled_imlist)==0:
            to_df.append([imname,path])
        elif imname in labeled_imlist:
            to_df.append([imname,path])

    df = pd.DataFrame(to_df,columns=['image_id','image_path'])
    imlist = df['image_id'].unique()

    return df, imlist

# function to convert a label dataframe to dictionary format
#   for use with mapcalc library to calculate mAP scores
#
def label_df_to_dict(df):
    
    # establish dictionary format
    label_dict = {"boxes":[], "labels":[]}

    # convert each row
    for index, row in df.iterrows():
        label_dict['boxes'].append([row['x1'],row['y1'],row['x2'],row['y2']])
        label_dict['labels'].append(row['label_bool'])

    # return dictionary
    return label_dict


# function to run the segmentation model
#   accepts a model and a list of image paths or directory containing them
#   returns a dataframe of images + paths and a list of unique image IDs
#   optionally accepts list of image IDs (for example, those with provided labels):
#       if len(list)>0, will only return image paths with matching IDs
#
def run_seg_model(model,images,verbose=False):
    # Initialize variables
    to_df = []
    error_ct = 0

    # Run the segmentation model
    results = model(images,stream=True)

    # Extract the bounding boxes and image ids from results
    for res in results:
        error_flag = False
        try:
            # if list of detected boxes has at least one element
            xyxy = torch.Tensor.tolist(res.boxes.xyxy)[0]
        except IndexError:
            # optionally output which image has the error
            if verbose:
                print('Index error with:',res.path.split('\\')[-1],
                      'nboxes = ',len(torch.Tensor.tolist(res.boxes.xyxy)))
            # add to count of segmentation errors
            error_ct += 1
            error_flag = True
            
            # load the image and label a center image as dummy
            img = cv2.imread(res.path,cv2.IMREAD_GRAYSCALE)
            h = img.shape[0]
            w = img.shape[1]
            xyxy = [0,0,w,h]

        im_name = res.path.split('\\')[-1].replace('.jpg','')
        to_df.append([im_name,xyxy[0],xyxy[1],xyxy[2],xyxy[3]])

    df = pd.DataFrame(to_df,columns=['image_id','x1','y1','x2','y2'])
    print("Segmentation error count:",error_ct)

    return df

#####################################################################
######## Required elements for DenseNet classification model ########

# Transformers for image augmentation using the albumentations module
#   not working on my machine yet

train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        # A.SmallestMaxSize(max_size=320),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        # A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.SmallestMaxSize(max_size=128),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        # A.SmallestMaxSize(max_size=320),
        # A.CenterCrop(height=256, width=256),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.SmallestMaxSize(max_size=128),
        ToTensorV2(),
    ]
)

# class for data used in classification dataloader
#
class PredData(torch.utils.data.Dataset):
    def __init__(self,df,folder_path,transform=None,verbose=False):
        self.df = df
        self.image_names = glob(folder_path+'\\**\\*.jpg',recursive=True)
        self.verbose = verbose

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,idx):
        image_name = self.image_names[idx]
        image_name_base = image_name.split('\\')[-1].replace('.jpg','')
        row = self.df[self.df['image_id']==image_name_base]
        left = int(round(row['x1'].item(),0))
        upper = int(round(row['y1'].item(),0))
        right = int(round(row['x2'].item(),0))
        lower = int(round(row['y2'].item(),0))

        label = row['label_bool'].item()

        # resize_factor = 320
        # resize_factor = 256
        resize_factor = 160

        # open the image using CV2 and enhance
        img = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
        img_eq = rayleigh_clahe(img)
        img_int = lanczos_interpolation(img_eq, 1)
        img_cfr = img_int
        final_img = Image.fromarray(img_cfr).convert('RGB').crop((left,upper,right,lower)).resize((resize_factor,resize_factor))

        # img = Image.open(root_path+folder+image_name).convert('RGB').crop((left,upper,right,lower))
        img_transformed = val_transform(image=np.array(final_img))['image'] 
        #img_enhanced = cleanup_image(image_name,path=True,resize=True
        #                             ,crop=True,crop_xyxy=(left,upper,right,lower),targetsize=(160,160))
        if self.verbose:
            print(image_name,'label=',label)
            #plt.imshow(img)
            #plt.show()

        #return T.ToTensor()(img)
        # return T.ToTensor()(img), label
        #return T.ToTensor()(img_enhanced), label
        #return img, label
        return img_transformed, label
    
# function to create a dataloader object for use by pytorch models
#   Utilizes the PredData class in this module
#   accepts a dataframe of image names and a folder path where images are located
#   returns the DataLoader object
#
def make_dataloader(df,image_path,verbose=False):
    
    dl = torch.utils.data.DataLoader(PredData(df,image_path,transform=False,verbose=verbose),
                                    batch_size = 16,
                                    shuffle = False,
                                    pin_memory = True if torch.cuda.is_available() else False
                                    )
    return dl

# function to run the classification model
#   accepts a dataloader and a model
#   returns tensors of predictions and true label classes
#
def predict_classes(dl,model,device,verbose=False):

    # Initialize variables
    pred_classes = []
    true_classes = []
    n_batches = 0
    n_inputs = 0

    # loop through each image and label pair using the dataloader
    for inputs, labels in dl:
    #for inputs in dl:
        n_batches += 1
        n_inputs += len(inputs)
        print("# batches / inputs:",n_batches,n_inputs)
        with torch.no_grad():
            true_classes += labels.tolist()
            inputs, labels = inputs.to(device),labels.to(device)
            #inputs = inputs.to(device)
            logps = model.forward(inputs).detach()
            top_p, top_class = logps.topk(1, dim=1)
            pred_classes += top_class.flatten().tolist()
            if verbose:
                print([(a,b) for a,b in zip(true_classes,pred_classes)])

        preds = torch.Tensor(pred_classes)
        true_y = torch.Tensor(true_classes)

    return preds, true_y
    #return preds

# function to accept tensors of predicted & actual binary class
#   using torchmetrics module
#   returns a dictionary of metrics
#
def score_classes(preds,true_y):

    # Initialize variables
    scores = {}

    # Record various metrics
    scores['Accuracy'] = round(tm.functional.accuracy(preds, true_y, task="binary").item(),3)
    scores['Precision'] = round(tm.functional.precision(preds,true_y,task="binary").item(),3)
    scores['Recall'] = round(tm.functional.recall(preds,true_y,task="binary").item(),3)
    scores['F1'] = round(tm.functional.f1_score(preds,true_y,task="binary").item(),3)
    #scores['DSC'] = tm.functional.dice(preds,true_y).item()
    scores['ConfMat'] = tm.functional.confusion_matrix(preds,true_y,task="binary")

    return scores