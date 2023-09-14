
from  cnn import CNN
import ast
import argparse
import time
 
output_file_path = 'data/augmented/successful_params_cnn.txt'


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
    params = [args.name, args.aug, int(args.percent), int(args.example)]
    print(params)
    if params not in successful_params:
       
        print(f'Running {args.name} dataset')
        train_path = f'data/augmented/{args.name}/{args.aug}/meth_{args.aug}_{args.percent}_pctwts_0.1_example_{args.example}.csv'
        test_path = f'data/augmented/{args.name}/{args.aug}/test.csv'

        
        w2v_path = 'w2v.pkl'
        dataset_name = args.name
        aug_method = args.aug
        num_example = int(args.example)
        percentage = int(args.percent)
        max_seq_len = 128
        batch_size = 4
        epochs = 20

        cnn = CNN(dims=300, w2v_path=w2v_path, aug_method = aug_method, percentage = percentage, num_example=num_example,fulldataset= False, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, chunk_size=1000)
        train_dataset, test_dataset, val_dataset, n_classes = cnn.insert_values(train_path, test_path)  # Updated to return datasets
        hist_dict, res_dict, avg_dict = cnn.run_n_times(train_dataset, test_dataset, val_dataset,  args.name, n=3)  # Updated to use datasets

        print('---------------------------------------------------')
        print(f'Average results for {args.name} dataset')
        print(avg_dict)
        print('---------------------------------------------------')
        with open(output_file_path, 'a') as output_file:
            formatted_params = f"('{params[0]}', '{params[1]}', {params[2]}, {params[3]}),"
            output_file.write(formatted_params + '\n')

            
        successful_params.append(params) 




