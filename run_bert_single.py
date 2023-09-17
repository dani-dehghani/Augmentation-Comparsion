
from simple_bert_ntimes import SimpleBert
import numpy as np
import ast
import argparse

 
output_file_path = 'data/augmented/successful_params_bert.txt'


successful_params = []
with open(output_file_path, 'r') as file:
    for line in file:
        row = line.strip(',\n"')
        #print(row)
        row = ast.literal_eval(row)
        successful_params.append(row)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('project')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--aug', type=str, default=None)
    parser.add_argument('--percent', type=str, default=None)
    parser.add_argument('--example', type=str, default=None)
    args = parser.parse_args()
    #print('started')
    #time.sleep(10)
    #print('finished')
    params = (args.name, args.aug, int(args.percent), int(args.example))
    print(params)
    if params not in successful_params:

        print(f'Running {args.name} dataset')
        train_path = f'data/augmented/{args.name}/{args.aug}/meth_{args.aug}_{args.percent}_pctwts_0.1_example_{args.example}.csv'
        test_path = f'data/augmented/{args.name}/{args.aug}/test.csv'

        dataset_name = args.name
        aug_method = args.aug
        num_example = int(args.example)
        percentage = int(args.percent)


        simple_bert = SimpleBert(dataset_name= dataset_name,aug_method=aug_method, percentage=percentage, num_example=num_example, fulldataset=False)
        print(f"Loaded data for {args.name} dataset")
        simple_bert.load_data(train_path,test_path )
        print(f"Trained model for {args.name} dataset")
        res = simple_bert.run_n_times(n=3)
        simple_bert.clean_up()


        params = [args.name, args.aug, int(args.percent), int(args.example)]
        with open(output_file_path, 'a') as output_file:
            formatted_params = f"('{params[0]}', '{params[1]}', {params[2]}, {params[3]}),"
            output_file.write(formatted_params + '\n')

                
        successful_params.append(params) 
        



