import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
def create_label_csv(args):
    save_path = '/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/new-splits-csvs-' + args.label + '/'
    if args.drop_cc:
        save_path = save_path + 'MLO_only/'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    if args.binary_birads:
        save_path += 'binary_'
    rsna_train_folds = '/mnt/storage/Devam/mammo-clip-github/Mammo-CLIP/src/codebase/data_csv/train_folds.csv'
    df = pd.read_csv(rsna_train_folds)
    path_concat_csv = '/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/rsna_stitched_images-flipped.csv'
    df_bilat = pd.read_csv(path_concat_csv)
    ## Adding fold information
    df_bilat['fold'] = 1
    ## Applying same folds as Mammo-CLIP csv for RSNA.
    df_bilat['fold'] = df_bilat['patient_id'].apply(lambda x: df[df['patient_id'] == x]['fold'].iloc[0])

    ### Distribution of BIRADS and FOLDS :
    print(df_bilat[args.label].value_counts())

    ### Merging the BIRADS 1,2 as 1
    if args.binary_birads:
        df_bilat[args.label] = df_bilat[args.label].apply(lambda x: 1 if x == 2 else x)
        print(f' distribution after merging {df_bilat[args.label].value_counts()}')

    
    print("dist before dropping : \n"  , df_bilat['fold'].value_counts(), "\n after dropping \n", df_bilat.dropna(subset=[args.label])['fold'].value_counts())

    #### Drop and split data frame

    # Split the data into train, validation, and test sets (.8,.1,.1)
    # Drop rows where label is NaN

    df_bilat_clean = df_bilat.dropna(subset=[args.label])

    # Get unique patient IDs
    unique_patient_ids = df_bilat_clean['patient_id'].unique()

    # Calculate sizes for train, validation, and test sets
    total_patients = len(unique_patient_ids)
    train_size = int(0.8 * total_patients)
    val_size = int(0.1 * total_patients)
    test_size = total_patients - train_size - val_size  # The rest will go to test

    # Split patient IDs into train, validation, and test sets
    train_patient_ids, remaining_patient_ids = train_test_split(unique_patient_ids, train_size=train_size, random_state=42)
    test_patient_ids, val_patient_ids = train_test_split(remaining_patient_ids, test_size=test_size, random_state=42)

    # Assign folds based on patient IDs
    train_df = df_bilat_clean[df_bilat_clean['patient_id'].isin(train_patient_ids)].copy()
    val_df = df_bilat_clean[df_bilat_clean['patient_id'].isin(val_patient_ids)].copy()
    test_df = df_bilat_clean[df_bilat_clean['patient_id'].isin(test_patient_ids)].copy()

    # Assign new folds
    train_df['fold'] = 1
    val_df['fold'] = 0
    test_df['fold'] = 2

    df_train = pd.concat([train_df,val_df])

    # Output value counts to verify
    print(val_df['fold'].value_counts())
    print(test_df['fold'].value_counts())
    print(train_df['fold'].value_counts())

    # Check for patient ID leakage
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])

    # leaked_patients = train_patients.intersection(val_patients, test_patients)
    # print(f"Number of leaked patients: {len(leaked_patients)}")

    leaked_patients = train_patients.intersection(val_patients).union(train_patients.intersection(test_patients)).union(val_patients.intersection(test_patients))
    print(f"Number of leaked patients: {len(leaked_patients)}")
    
    if args.drop_cc:
        df_train = df_train[df_train['view'] == 'MLO']
        test_df = test_df[test_df['view'] == 'MLO']

    print("Train set shape:", train_df.shape[0])
    print("Validation set shape:", val_df.shape[0])
    print("Test set shape:", test_df.shape[0])
    print("Train DF : " , df_train.head(10))
    if args.save :
        df_train.to_csv(save_path + 'train_folds_rsnalp_final.csv', index=False)
        test_df.to_csv( save_path + 'test_folds_rsnalp_final.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_birads', action='store_true', help='Flag to indicate binary BIRADS')
    parser.add_argument('--label', type=str, default='BIRADS', help='Label to use for classification')
    parser.add_argument('--save', action='store_true', help='Flag to save the CSVs')
    parser.add_argument('--save_path', type=str, default='/mnt/storage/Devam/mammo-clip-github/image_preprocessing-aisha/new-splits-csvs-birads/', help='Path to save the CSVs')
    parser.add_argument('--drop_cc', action='store_true', help='Flag to indicate dropping MLO images')
    args = parser.parse_args()

    create_label_csv(args)