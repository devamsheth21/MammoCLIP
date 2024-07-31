import pandas as pd
import numpy as np
import os
import pickle as pkl
import re
import cv2
from skimage import io
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure as skmeas


def check_laterality(x, laterality):
    correct_laterality = True
    mid = x.shape[1]//2
    if laterality == 'R':
        if np.mean(x[:,0:mid]) > np.mean(x[:,-mid:-1]):  # x[:,0] is the far left
            laterality = 'L'
            correct_laterality = False

    elif laterality == 'L':
        if np.mean(x[:,0:mid]) < np.mean(x[:,-mid:-1]):
            laterality = 'R'
            correct_laterality = False

    return correct_laterality


def main():
    df = pd.read_csv('/mnt/storage/breast_cancer_kaggle/train.csv')
    img_dir = '/mnt/storage/breast_cancer_kaggle/mammo_clip/train_images_png/'
    df.loc[:, 'incorrect_laterality'] = False
    for idx in tqdm(range(len(df))):
        patient_id = df.at[idx, 'patient_id']
        image_id = df.at[idx, 'image_id']
        view = df.at[idx, 'view']
        laterality = df.at[idx, 'laterality']
        img_path = os.path.join(img_dir, str(patient_id), str(image_id) + '.png')
        img = cv2.imread(img_path)

        if not check_laterality(img, laterality):
            updated_laterality = 'R' if laterality == 'L' else 'L'
            # print(f'Patient ID: {patient_id}, Image ID: {image_id}, View: {view}, Original Laterality: {laterality}, Updated Laterality: {updated_laterality}')
            df.at[idx, 'incorrect_laterality'] = True
    df.to_csv('/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/train_corrected.csv', index=False)
    patients_updated = df[df['incorrect_laterality']==True]['patient_id'].unique()
    print("Number of updated laterality: ",len(patients_updated))
    print("list of patients updated: ",patients_updated)
    print("List of image ids updated: ",df[df['incorrect_laterality']==True]['image_id'].unique())

if __name__ == '__main__':
    main()