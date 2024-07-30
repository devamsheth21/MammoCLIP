import pandas as pd
import ast
import argparse

def generate_description(row):
    density_mapping = {
        'A': 'Almost entirely fatty: The breasts are composed of mostly fatty tissue.',
        'B': 'Scattered fibroglandular densities: There are scattered areas of dense fibroglandular tissue.',
        'C': 'Heterogeneously dense: The breasts have heterogeneously dense tissue, which may obscure small masses.',
        'D': 'Extremely dense: The breasts are extremely dense, which may lower the sensitivity of mammography.'
    }
    BIRADS = row['BIRADS']
    BIRADS_Assessment_Description = {
        0: 'Incomplete: Need additional imaging evaluation.',
        1: 'Negative: No mammographic evidence of malignancy.',
        2: 'Benign: No mammographic evidence of malignancy.',
        3: 'Probably benign: <2% risk of malignancy.',
        4: 'Suspicious: 2-94% risk of malignancy.',
        5: 'Highly suggestive of malignancy: >95% risk of malignancy.',
        6: 'Known biopsy-proven malignancy.'
    }.get(BIRADS, '[Not provided]')
    cancer = 'Cancer present' if row['cancer'] == 1 else 'No cancer present'
    biopsy = 'Biopsy performed' if row['biopsy'] == 1 else 'No biopsy performed'
    invasive = 'Invasive cancer' if row['invasive'] == 1 else 'Non-invasive cancer'
    density_description = density_mapping.get(row['density'], '[Not provided]')
    difficult_negative_case = 'A difficult negative case' if row['difficult_negative_case'] == 1 else ''
    
    # Fill None values with a specified value
    laterality = row['laterality'] if row['laterality'] is not None else '[Not provided]'
    view = row['view'] if row['view'] is not None else '[Not provided]'
    age = row['age'] if row['age'] is not None else '[Not provided]'
    
    report = f"""Procedure Reported: \nDigital mammography, both breasts. Laterality: {laterality}, View: {view} .\n
Breast Composition: {density_description} \n
Patient Age: {age}  \n
Impression:
{cancer} 
{biopsy} 
{invasive}
{difficult_negative_case}
BI-RADS Assessment:
- Category {BIRADS}: {BIRADS_Assessment_Description} """
    report = report.replace('\n', " ")
    return report

def filter_by_view(group, view_type):
    return [group[group['view'] == view_type]['image_id'].tolist()]

def reorder_sequence(row):
    view_index_map = {'CC': 0, 'MLO': 1}
    new_order = ['CC', 'MLO']
    new_image = [None, None]

    for i, view in enumerate(row['view']):
        index = view_index_map.get(view)
        if index is not None:
            new_image[index] = row['image'][i]

    row['view'] = new_order
    row['image'] = new_image

    return row

def main(input_path, output_path):
    # input_path = '/mnt/storage/Devam/mammo-clip-github/Mammo-CLIP/src/codebase/data_csv/train_folds.csv'
    df = pd.read_csv(input_path)
    df.dropna(subset=['BIRADS'], inplace=True)
    print(df.columns)
    df['text'] = df.apply(generate_description, axis=1)
    # df['age'] = df['age'].astype(int)
    new_df = df.groupby(['patient_id', 'laterality']).apply(lambda group: pd.Series({  
        'cc_image_paths': filter_by_view(group, 'CC')[0],
        'mlo_image_paths': filter_by_view(group, 'MLO')[0],
        'image_ids': group['image_id'].tolist(),
        'views': group['view'].tolist(),
        'fold': group['fold'].iloc[0],
        'text': group['text'].tolist(),
        'cancer': group['cancer'].iloc[0],
        'biopsy': group['biopsy'].iloc[0],
        'invasive': group['invasive'].iloc[0],
        'density': group['density'].iloc[0],
        'age': group['age'].iloc[0],
        'birads': group['BIRADS'].iloc[0],
        'difficult_negative_case': group['difficult_negative_case'].iloc[0]
    })).reset_index()
    new_df.rename(columns={
        'image_ids': 'image', 
        'cc_image_paths': 'CC',
        'mlo_image_paths': 'MLO',
        'views': 'view',
    }, inplace=True)
    new_df['patient_id_laterality'] = new_df['patient_id'].astype(str) + '_' + new_df['laterality']
    new_df['view'] = new_df['view'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    new_df['image'] = new_df['image'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_final = new_df.apply(reorder_sequence, axis=1)
    # output_path = '/mnt/storage/Devam/mammo-clip-github/Mammo-CLIP/src/codebase/data_csv/train_folds_final.csv'
    df_final.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RSNA Preprocessing')
    parser.add_argument('--input_path', type=str, default='/mnt/storage/Devam/mammo-clip-github/Mammo-CLIP/src/codebase/data_csv/train_folds.csv', help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, default='/mnt/storage/Devam/mammo-clip-github/Mammo-CLIP/src/codebase/data_csv/new_rsna.csv', help='Path to the output CSV file')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    main(input_path, output_path)
