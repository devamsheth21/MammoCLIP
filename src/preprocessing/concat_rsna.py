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

def stitch_images(x, y):
    diff = abs(x.shape[0] - y.shape[0])
    if x.shape[0] < y.shape[0]:
        z = np.zeros((diff, x.shape[1], x.shape[2]))
        x = np.concatenate((x, z), axis=0)
    else:
        if x.shape[0] > y.shape[0]:
            z = np.zeros((diff, y.shape[1], y.shape[2]))
            y = np.concatenate((y, z), axis=0)
    img = np.concatenate((x, y), axis=1)

    if img.shape[1] > img.shape[0]:
        # print('wide')
        diff = img.shape[1] - img.shape[0]
        z = np.zeros((diff, img.shape[1], img.shape[2]))
        img = np.concatenate((img, z), axis=0)
        img = np.array(img, dtype='uint8')
    elif img.shape[0] > img.shape[1]:
        print('tall')
        diff = img.shape[0] - img.shape[1]
        z1 = np.zeros((img.shape[0], np.int(diff/2), img.shape[2]))
        z2 = np.zeros((img.shape[0], diff - np.int(diff/2), img.shape[2]))
        img = np.concatenate((z2, img, z1), axis=1)
        img = np.array(img, dtype='uint8')
    else:
        print('square')

    return img


def main():
    df = pd.read_csv('/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/train_corrected.csv')

    img_dir = '/mnt/storage/breast_cancer_kaggle/mammo_clip/train_images_png/'
    output_folder = '/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/rsna-stitched-images-orgdim-flipped'
    output_folder_resized = '/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/rsna-stitched-images-512-flipped'
    df_new = pd.DataFrame(columns=['site_id', 'patient_id', 'image_id', 'view', 'age', 'cancer', 'biopsy', 'invasive', 'BIRADS', 'implant', 'density', 'machine_id', 'difficult_negative_case'])

    df_updated = df[(df['view'] == 'MLO') | (df['view'] == 'CC')]
    # df_updated = df_updated[df['incorrect_laterality']== True]
    df_updated['laterality'] = df_updated.apply(lambda x: 'R' if x['incorrect_laterality'] == True and x['laterality'] == 'L' else ('L' if x['incorrect_laterality'] == True and x['laterality'] == 'R' else x['laterality']), axis=1)

    grouped_df = df_updated.groupby(['patient_id', 'view'])

    for i, ((patient_id, view), patient_rows) in tqdm(enumerate(grouped_df)):
        lateralities = patient_rows['laterality'].unique()
        if 'L' in lateralities and 'R' in lateralities:
            image_ids_L = patient_rows[patient_rows['laterality'] == 'L']['image_id'].values
            image_ids_R = patient_rows[patient_rows['laterality'] == 'R']['image_id'].values

            image_combinations = [(image_id_L, image_id_R) for image_id_L in image_ids_L for image_id_R in image_ids_R]
            for image_id_L, image_id_R in image_combinations:
                img_L_path = os.path.join(img_dir, str(patient_id), str(image_id_L) + '.png')
                img_R_path = os.path.join(img_dir, str(patient_id), str(image_id_R) + '.png')

                img_L = cv2.imread(img_L_path)
                img_R = cv2.imread(img_R_path)
                


                stitched_img = stitch_images(img_R, img_L)

                stitched_image_id = str(image_id_L) + '_' + str(image_id_R)

                birads_values = patient_rows[patient_rows['image_id'].isin([image_id_L, image_id_R])]['BIRADS'].values
                cancer_values = patient_rows[patient_rows['image_id'].isin([image_id_L, image_id_R])]['cancer'].values
                density_values = patient_rows[patient_rows['image_id'].isin([image_id_L, image_id_R])]['density'].values
                if pd.isna(birads_values[0]):
                    birads = birads_values[1]
                elif pd.isna(birads_values[1]):
                    birads = birads_values[0]
                else:
                    if birads_values[1] > 0 and birads_values[0] > birads_values[1]:
                        birads = birads_values[0]
                    else:
                        birads = birads_values[1]
                cancer = cancer_values[0] if cancer_values[0] > cancer_values[1] else cancer_values[1]
                density = density_values[0] if density_values[0] else density_values[1]

                stitched_img_path = os.path.join(output_folder, str(patient_id), str(stitched_image_id) + '.png')
                resized_img_path = os.path.join(output_folder_resized, str(patient_id), str(stitched_image_id) + '.png')

                os.makedirs(os.path.dirname(stitched_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(resized_img_path), exist_ok=True)
                io.imsave(stitched_img_path, np.array(stitched_img, dtype='uint8'))
                resized_img = cv2.resize(stitched_img, (512, 512))
                io.imsave(resized_img_path, np.array(resized_img, dtype='uint8'))

                df_new = df_new._append({'site_id': patient_rows['site_id'].values[0],
                                        'patient_id': patient_rows['patient_id'].values[0],
                                        'image_id': stitched_image_id,
                                        'view': patient_rows['view'].values[0],
                                        'age': patient_rows['age'].values[0],
                                        'cancer': cancer,
                                        'biopsy': patient_rows['biopsy'].values[0],
                                        'invasive': patient_rows['invasive'].values[0],
                                        'BIRADS': birads,
                                        'implant': patient_rows['implant'].values[0],
                                        'density': density,
                                        'machine_id': patient_rows['machine_id'].values[0],
                                        'difficult_negative_case': patient_rows['difficult_negative_case'].values[0]}, ignore_index=True)

    df_new.to_csv('/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/rsna_stitched_images-flipped.csv', index=False)


if __name__ == "__main__":
    main()
