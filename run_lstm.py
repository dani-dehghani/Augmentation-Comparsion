
from itertools import product
from lstm import *
import ast

dataset_list1 = ['agnews', 'subj', 'pc', 'yelp', 'cr']
dataset_list2 = [  'sst2', 'pubmed', 'trec','cardio','bbc']
aug_list = ['aeda', 'backtranslation', 'charswap', 'checklist', 'clare', 'deletion', 'eda', 'embedding', 'wordnet']
aug_percent = [10, 20, 50, 100]
example_list = [1, 2, 4]

# list of all possible combinations
param_combinations = list(product(dataset_list2, aug_list, aug_percent, example_list))

 
output_file_path = 'data/augmented/successful_params_lstm.txt'


successful_params = []
with open(output_file_path, 'r') as file:
    for line in file:
        row = line.strip(',\n"')
        print(row)
        row = ast.literal_eval(row)
        successful_params.append(row)

if __name__ == '__main__':
    for params in param_combinations:
        if params not in successful_params:
          
            name, aug, percent, i = params
            print(f'Running {name} dataset')
            train_path = f'data/augmented/{name}/{aug}/meth_{aug}_{percent}_pctwts_0.1_example_{i}.csv'
            test_path = f'data/augmented/{name}/{aug}/test.csv'
    
            
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
    
            with open(output_file_path, 'a') as output_file:
                formatted_params = f"('{params[0]}', '{params[1]}', {params[2]}, {params[3]}),"
                output_file.write(formatted_params + '\n')

             
            successful_params.append(params) 


