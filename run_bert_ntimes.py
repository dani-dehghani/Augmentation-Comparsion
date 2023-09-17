from simple_bert_ntimes import SimpleBert
import numpy as np
from itertools import product
import ast

np.random.seed(100)
dataset_list = ['cr', 'trec', 'agnews', 'pc', 'yelp', 'sst2','subj','cardio', 'bbc']

aug_list = ['aeda', 'backtranslation', 'charswap', 'checklist', 'clare', 'deletion', 'eda', 'embedding', 'wordnet']

aug_percent = [10, 20, 50, 100]

example_list = [1, 2, 4]


# list of all possible combinations
param_combinations = list(product(dataset_list, aug_list, aug_percent, example_list))

 
output_file_path = 'data/augmented/successful_params_bert.txt'


successful_params = []
with open(output_file_path, 'r') as file:
    for line in file:
        row = line.strip(',\n"')
        print(row)
        row = ast.literal_eval(row)
        successful_params.append(row)

if __name__ == "__main__":
    for params in param_combinations:
        if params not in successful_params:   
    
    

            name, aug, percent, i = params
            print(f'Running {name} dataset')
            train_path = f'data/augmented/{name}/{aug}/meth_{aug}_{percent}_pctwts_0.1_example_{i}.csv'
            test_path = f'data/augmented/{name}/{aug}/test.csv'
            aug_method = aug
            num_example = i
            percentage = percent
            simple_bert = SimpleBert(dataset = name,aug_method=aug_method,percentage=percentage, num_example=num_example,fulldataset= False)
            print(f"Loaded data for {name} dataset")
            simple_bert.load_data(train_path,test_path )
            print(f"Trained model for {name} dataset")
            res = simple_bert.run_n_times(n=3)
            # simple_bert.train_model()
            # print(f"Evaluated model for {dataset} dataset")
            # res = simple_bert.evaluate_model()                
            # model will be saved during the training process
            #simple_bert.save_results(f"results/augmented/bert/full/{dataset}_full_results.txt",write_to_file=True,n_times=True)              # change the aug or original folder here
            #print(f"Saved model and results for {name} dataset")
            #print('cleaning up the checkpoint folders')
            simple_bert.clean_up()
            print(f'results: \n\n\n{res}\n\n\n')
            with open(output_file_path, 'a') as output_file:
                formatted_params = f"('{params[0]}', '{params[1]}', {params[2]}, {params[3]}),"
                output_file.write(formatted_params + '\n')

             
            successful_params.append(params) 




        #pre_last_layer_output = simple_bert.extract_pre_last_layer(text)
        #print(pre_last_layer_output)