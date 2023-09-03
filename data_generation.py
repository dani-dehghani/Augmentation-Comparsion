#%%
import pandas as pd
import os
import random

#%%

dataset_list = ['agnews','subj','pc','yelp','cr','cardio','bbc','sst2','pubmed','trec']
aug_list = ['aeda', 'backtranslation', 'charswap', 'checklist', 'clare', 'deletion','eda', 'embedding', 'wordnet']
aug_percent = [10,20,50]
example_list = [1,2]
#%%
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

#%%
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

def augmented_example(dataset, aug_method, sampling_percentage, example=4):
    data_path = f'data/original/{dataset}'
    original_df = pd.read_csv(os.path.join(data_path, f'train_{sampling_percentage}_percent.csv'))
    #original_df = pd.read_csv(os.path.join(data_path, 'train.csv'))

    augmented_path = f'data/augmented/{dataset}/{aug_method}'  # Corrected variable name
    augmented_df = pd.read_csv(os.path.join(augmented_path, f'meth_{aug_method}_pctwts_0.1_example_4.csv'))

    # Find rows in the augmented dataset that are also in the original dataset
    common_rows = augmented_df[augmented_df.isin(original_df.to_dict(orient='list')).all(1)]

    # Extract the indices of common rows and the next 4 rows
    indices_to_extract = []
    for index in common_rows.index:
        if index + example < len(augmented_df):
            indices_to_extract.extend(range(index, index + example + 1))

    # Remove duplicates and create a new DataFrame
    indices_to_extract = list(set(indices_to_extract))
    df = augmented_df.iloc[indices_to_extract]

    # Define a file path for the output CSV file
    output_csv_path = os.path.join(augmented_path, f'meth_{aug_method}_{sampling_percentage}_pctwts_0.1_example_{example}.csv')
    #output_csv_path = os.path.join(augmented_path, f'meth_aeda_pctwts_0.1_example_{example}.csv')

    df.to_csv(output_csv_path, index=False)

#%%


if __name__ == '__main__':
    for name in dataset_list:
        for aug_method in aug_list:
            for percent in aug_percent:
                for i in example_list:
                    augmented_example(name,aug_method, percent, i)















