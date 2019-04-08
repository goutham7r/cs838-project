import csv
import cv2
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm

import urllib
import cv2, csv
from matplotlib import pyplot as plt
 
    
import torch, torchvision
import torch.utils.data
import pandas as pd
from torch.utils.data.dataset import Dataset  
from torchvision import transforms, utils



ASPECT_LB = 0.75
ASPECT_UB = 1.25

    
def url_to_image(url):
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except:
        return None

    
def check_and_resize(image):
    if ASPECT_LB < image.shape[0]/image.shape[1] < ASPECT_UB:
        return cv2.resize(image, (128, 128))
    else:
        return None
   

def get_celeb_list():
    
    data_info = {}
    with open("../../IMDb-Face.csv", "r", encoding="utf8") as f:
        l = sum(1 for line in f)
    
    with open("../../IMDb-Face.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in tqdm(enumerate(reader), total=l):
            try:
                bounds = [int(x) for x in line[3].split()]
                w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
                if ASPECT_LB < w/h < ASPECT_UB:
                    if line[0] in data_info:
                        data_info[line[0]] += 1
                    else:
                        data_info[line[0]] = 1
            except:
                pass

    freq = {}
    for key in data_info:
        value = data_info[key]
        if value in freq:
            freq[value] += 1
        else:
            freq[value] = 1

    celebs = [key for key in data_info if data_info[key]>120] # change this cutoff if necessary
#     celebs = celebs[:num_celebs]
    return celebs

def fetch_images(celebs, prefix="", num_train=80, num_val=10, num_test=10, num_celebs=750):
    
    with open("../IMDb-Face.csv", "r", encoding="utf8") as f:
        l = sum(1 for line in f)
    
    num_imgs = num_train + num_val + num_test
    output_dir = "images"
    dirs= [output_dir, os.path.join(output_dir, "train"), os.path.join(output_dir, "val"), os.path.join(output_dir, "test")]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    done = {}
    complete = 0

    count = 1
    

    with open("../IMDb-Face.csv", "r", encoding="utf8") as f:
        f_train = open(os.path.join(output_dir,"train_labels.csv"), mode='w+', newline="\n", encoding="utf-8")
        f_val = open(os.path.join(output_dir,"val_labels.csv"), mode='w+', newline="\n", encoding="utf-8")
        f_test = open(os.path.join(output_dir,"test_labels.csv"), mode='w+', newline="\n", encoding="utf-8")
        
        writer_train = csv.writer(f_train)
        writer_train.writerow(["Filename","Celeb Name", "Label"])
        
        writer_val = csv.writer(f_val)
        writer_val.writerow(["Filename","Celeb Name", "Label"])
        
        writer_test = csv.writer(f_test)
        writer_test.writerow(["Filename","Celeb Name", "Label"])

        reader = csv.reader(f, delimiter=",")
        for i, line in tqdm(enumerate(reader), total=l):
            if i==0:
                continue
            celeb_name = line[0]
            
#             t = {k: v for k, v in done.items() if v==num_imgs}
#             if len(t)==num_celebs:
#                 break

            if celeb_name not in celebs:
                continue

            if celeb_name not in done:
                done[celeb_name] = 0
                writer = writer_train
                mid_dir = "train"
                q = []

            if done[celeb_name]==num_train:
                writer = writer_val
                mid_dir = "val"
            
            if done[celeb_name]==num_train+num_val:
                writer = writer_test
                mid_dir = "test"
            
            if done[celeb_name]==num_imgs:
                if q is not None:
                    print("\n",celeb_name,": ", end='')
                    for i in range(num_train):
                        ct = complete*num_train + i + 1
                        image, celeb_name, label = q[i]
                        img_filename = os.path.join(output_dir, "train", prefix+"%d.jpg"%ct)
                        plt.imsave(img_filename, image)
                        writer_train.writerow([img_filename, celeb_name, len(done)-1])
#                         print("|", end='')
                    for i in range(num_val):
                        ct = complete*num_val + i + 1
                        image, celeb_name, label = q[i+num_train]
                        img_filename = os.path.join(output_dir, "val", prefix+"%d.jpg"%ct)
                        plt.imsave(img_filename, image)
                        writer_val.writerow([img_filename, celeb_name, len(done)-1])
#                         print("|", end='')
                    for i in range(num_test):
                        ct = complete*num_test + i + 1
                        image, celeb_name, label = q[i+num_train+num_val]
                        img_filename = os.path.join(output_dir, "test", prefix+"%d.jpg"%ct)
                        plt.imsave(img_filename, image)
                        writer_test.writerow([img_filename, celeb_name, len(done)-1])
#                         print("|", end='')
                    complete += 1
                    q = None
                    if complete==num_celebs:
                        break
                else:
                    continue
            
            
            img_filename = os.path.join(output_dir, mid_dir, "%d.jpg"%count)
    #         faces_img_dir = os.path.join(output_dir, "Faces", celeb_name)
    #         full_img_dir = os.path.join(output_dir,"Full", celeb_name)
    #         if not os.path.exists(faces_img_dir):
    #             os.makedirs(faces_img_dir)
    #         if not os.path.exists(full_img_dir):
    #             os.makedirs(full_img_dir)

            size = [int(x) for x in line[4].split()]
            bounds = [int(x) for x in line[3].split()]
            url = line[5]
            image = url_to_image(url)
            if image is not None:
                image = cv2.resize(image, (size[1], size[0]))
                image = image[:,:,::-1]
                try:
    #                 plt.imsave(os.path.join(full_img_dir, line[2]), image)
                    image = image[bounds[1]:bounds[3],bounds[0]:bounds[2],:]
                    image = check_and_resize(image)
                    if image is not None:
                        q.append((image, celeb_name, len(done)-1))
#                         plt.imsave(img_filename, image)
                        done[celeb_name] += 1
#                         print("|", end='')row
#                         writer.writerow([img_filename, celeb_name, len(done)-1])
                        count += 1
                except:
                    pass
        f_train.close()
        f_val.close()
        f_test.close()

    
    
class FacesDataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms=None):
        self.labels_frame = pd.read_csv(os.path.join(csv_file))
        self.root_dir = root_dir
        # self.transform = transforms.Compose([transforms.ColorJitter(),
        #                                      transforms.RandomAffine(degrees, translate=None, scale=None, shear=None,),
        #                                      ToTensor()])
        self.transform=transforms
        self.num_celebs = int(self.labels_frame.iloc[len(self.labels_frame)-1, 2])+1
        self.labels_frame.drop(0)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.labels_frame.iloc[idx, 0])
        image = plt.imread(img_name)
        
        def one_hot(idx):
            a = np.zeros(self.num_celebs)
            a[idx] = 1
            return a
        
        sample = (idx, (image, one_hot(self.labels_frame.iloc[idx, 2])))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    

    def __len__(self):
        return len(self.labels_frame)


    
    
    