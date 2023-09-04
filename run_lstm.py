from itertools import product
from lstm import *

dataset_list = ['agnews', 'subj', 'pc', 'yelp', 'cr', 'cardio', 'bbc', 'sst2', 'pubmed', 'trec']
aug_list = ['aeda', 'backtranslation', 'charswap', 'checklist', 'clare', 'deletion', 'eda', 'embedding', 'wordnet']
aug_percent = [10, 20, 50, 100]
example_list = [1,2,4]

# list of all possible combinations
param_combinations = list(product(dataset_list, aug_list, aug_percent, example_list))

if __name__ == '__main__':
    for params in param_combinations:
        name, aug, percent, i = params
        print(f'Running {name} dataset')
        train_path = f'data/augmented/{name}/{aug}/meth_{aug}_{percent}_pctwts_0.1_example_{i}.csv'
        test_path = f'data/augmented/{name}/{aug}/test.csv'
        
        try:
            w2v_path = 'w2v.pkl'
            dataset_name = name
            aug_method = aug
            num_example = i
            percentage = percent
            max_seq_len = 128
            batch_size = 16
            epochs = 20

            lstm = LSTM(dims=300, w2v_path=w2v_path, aug_method=aug_method, percentage=percentage, num_example=num_example, fulldataset=False, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs)
            train_dataset, test_dataset, val_dataset, n_classes = lstm.insert_values(train_path, test_path)
            hist_dict, res_dict, avg_dict = lstm.run_n_times(train_dataset, test_dataset, val_dataset, name, n=3)

            print('---------------------------------------------------')
            print(f'Average results for {name} dataset')
            print(avg_dict)
            print('---------------------------------------------------')
            
            # Remove the successfully completed experiment from the list of combinations
            param_combinations.remove(params)
            print(param_combinations)
        except Exception as e:
            print(f'Error in {name}: {str(e)}')
