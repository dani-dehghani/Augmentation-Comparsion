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