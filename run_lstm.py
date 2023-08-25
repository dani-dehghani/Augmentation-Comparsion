from lstm import *

#dataset_list = ['agnews','subj','pc','yelp','cr','kaggle_med','cardio','bbc','sst2']
dataset_list = ['trec']

if __name__ == '__main__':
    for name in dataset_list:
        #try:
            print (f'Running {name} dataset')
            train_path  = f'data/original/{name}/train_10.csv'
            test_path   = f'data/original/{name}/test.csv'
            w2v_path = 'w2v.pkl'
            dataset_name = f'{name}'
            max_seq_len = 128
            batch_size = 16
            epochs = 20
            lstm = LSTM(dims=300, w2v_path=w2v_path, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs)
            train_dataset, test_dataset, val_dataset, n_classes = lstm.insert_values(train_path, test_path)
            hist_dict, res_dict, avg_dict = lstm.run_n_times(train_dataset, test_dataset, val_dataset, name, n=3)
            
            
            test_data = [sample[0] for sample in test_dataset.as_numpy_iterator()]
            labels = [sample[1] for sample in test_dataset.as_numpy_iterator()]

            # Call visualize_tsne and extract_pre_last_layer with test_data
            pre_last_layer_features = lstm.extract_pre_last_layer(test_data[0:300])
            lstm.visualize_tsne(pre_last_layer_features, labels[0:300])
            print ('---------------------------------------------------')
            print (f'Average results for {name} dataset')
            print (avg_dict)
            print ('---------------------------------------------------')
        #except:
            #print (f'Error in {name}')
            #continue
