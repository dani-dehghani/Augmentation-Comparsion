#%%
import pandas as pd
import os
import random

#%%

dataset_list = ['agnews','subj','pc','yelp','cr','cardio','bbc','sst2','pubmed','trec']


import pandas as pd
import os

def data_percentage(dataset):
    print(f'{dataset} dataset')
    data_path = f'data/original/{dataset}'
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))

    # Split the DataFrame into 10%, 20%, and 50%
    train_10_percent = df.sample(frac=0.1, random_state=42)
    train_20_percent = df.sample(frac=0.2, random_state=42)
    train_50_percent = df.sample(frac=0.5, random_state=42)

    # Define file paths for the output CSV files
    train_10_percent_path = os.path.join(data_path, 'train_10_percent.csv')
    train_20_percent_path = os.path.join(data_path, 'train_20_percent.csv')
    train_50_percent_path = os.path.join(data_path, 'train_50_percent.csv')

    # Save the sampled DataFrames as separate CSV files
    train_10_percent.to_csv(train_10_percent_path, index=False)
    train_20_percent.to_csv(train_20_percent_path, index=False)
    train_50_percent.to_csv(train_50_percent_path, index=False)


def process_large_csv(testset, max_rows=200):
    data_path = f'data/original/{testset}'
    df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    # Check if the dataset is larger than 'max_rows'
    if len(df) > max_rows:
        # Randomly select 'max_rows' rows
        random_sample = df.sample(n=max_rows, random_state=42)

        # Define a file path for the output CSV file
        output_csv_path = os.path.join(data_path, 'test.csv')

        # Save the selected rows as a new CSV file
        random_sample.to_csv(output_csv_path, index=False)



#%%

if __name__ == '__main__':
    for name in dataset_list:
        data_percentage(name)
        process_large_csv(name)
        
#%%
def remove_na_augmentations(df_aug, df_org):
    """
    This function identifies original and augmented texts and removes NA augmentations.

    Parameters:
df_aug: DataFrame, the augmented data
df_org: DataFrame, the original data

    Returns:
df_aug: DataFrame, the augmented data with NAs removed
    """
    df_aug["org"] = df_aug["text"].isin(df_org["text"])
    df_aug['first_aug'] = df_aug['text'].shift(-1)
    df_aug['first_aug'] = df_aug['first_aug'].where(df_aug['first_aug'] != df_aug['text'], None)
    df_aug['second_aug'] = df_aug['text'].shift(-2)
    df_aug['second_aug'] = df_aug['second_aug'].where(df_aug['second_aug'] != df_aug['text'], None)
    na_indices = df_aug[df_aug['first_aug'].isna() | df_aug['second_aug'].isna()].index
    df_aug = df_aug.drop(na_indices)

    return df_aug


def select_org_and_aug_cols(df_aug):
    """
    This function selects relevant columns of original and augmented texts.

    Parameters:
df_aug: DataFrame, the augmented data

    Returns:
df_result: DataFrame, the selected columns from the data
    """
    df_result = df_aug[df_aug['org'] == True][['class', 'text', 'first_aug', 'second_aug']]
    return df_result


def create_augmentations(df_result, df_aug):
    """
    This function creates DataFrame of original and augmented texts.

    Parameters:
df_result: DataFrame, the selected columns from the data
df_aug: DataFrame, the augmented data

    Returns:
df_one_example: DataFrame, one example of augmented data
df_two_examples: DataFrame, two examples of augmented data
    """
    df_result = df_result[~df_result['first_aug'].isin(df_aug[df_aug['org'] == True]['text'])]
    df_result = df_result[~df_result['second_aug'].isin(df_aug[df_aug['org'] == True]['text'])]

    df_first_aug = df_result[['class', 'first_aug']].copy()
    df_first_aug.rename(columns={'first_aug': 'text'}, inplace=True)

    df_second_aug = df_result[['class', 'second_aug']].copy()
    df_second_aug.rename(columns={'second_aug': 'text'}, inplace=True)

    df_result['aug_number'] = 'original'
    df_first_aug['aug_number'] = 'first_aug'
    df_second_aug['aug_number'] = 'second_aug'

    df_all = pd.concat([df_result, df_first_aug, df_second_aug])

    df_all.sort_index(inplace=True)

    df_one_example = df_all[df_all['aug_number'].isin(['original', 'first_aug'])]
    df_two_examples = df_all[df_all['aug_number'].isin(['original', 'first_aug', 'second_aug'])]

    df_one_example = df_one_example[['class', 'text']]
    df_two_examples = df_two_examples[['class', 'text']]

    return df_one_example, df_two_examples


def create_aug_df_from_4_example(dataset_name,method_name):
    """
    This function creates a dataframe with one and two augmented versions from the original
    and four augmentations.

    Parameters:
dataset_name: str, name of the dataset
method_name: str, name of the method used for data augmentation

    Returns:
None. The function writes the output to CSV files.
    """
    aug_path = f'data/augmented/{dataset_name}/meth_{method_name}_pctwts_0.5_example_4.csv'
    org_path = f'data/original/{dataset_name}/train.csv'

    df_aug = pd.read_csv(aug_path)
    df_org = pd.read_csv(org_path)

    df_aug = remove_na_augmentations(df_aug, df_org)
    df_result = select_org_and_aug_cols(df_aug)

    df_one_example, df_two_examples = create_augmentations(df_result, df_aug)

    df_one_example.to_csv(f'data/augmented/{dataset_name}/meth_{method_name}_pctwts_0.5_example_1.csv', index=False)
    df_two_examples.to_csv(f'data/augmented/{dataset_name}/meth_{method_name}_pctwts_0.5_example_2.csv', index=False)






