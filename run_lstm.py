from lstm import *

#dataset_list = ['agnews','subj','pc','yelp','cr','cardio','bbc','sst2','pubmed','trec']
dataset_list = ['agnews']
aug_list = ['aeda', 'backtranslation', 'charswap', 'checklist', 'clare', 'deletion','eda', 'embedding', 'wordnet']
aug_percent = [10,20,50,100]
example_list = [1,2,4]
if __name__ == '__main__':
    for name in dataset_list:
        for aug in aug_list:
            for percent in aug_percent:
                for i in example_list:
                    #try:
                        print (f'Running {name} dataset')
                        #train_path  = f'data/original/{name}/train_50_percent.csv'
                        #test_path   = f'data/original/{name}/test.csv'
                        train_path  = f'data/augmented/{name}/{aug}/meth_{aug}_{percent}_pctwts_0.1_example_{i}.csv'
                        test_path   = f'data/augmented/{name}/{aug}/test.csv'
                        w2v_path = 'w2v.pkl'
                        dataset_name = f'{name}'
                        max_seq_len = 128
                        batch_size = 16
                        epochs = 20
                        lstm = LSTM(dims=300, w2v_path=w2v_path,aug_method=aug,percentage=percent, num_example=i,fulldataset= False, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs)
                        train_dataset, test_dataset, val_dataset, n_classes = lstm.insert_values(train_path, test_path)
                        hist_dict, res_dict, avg_dict = lstm.run_n_times(train_dataset, test_dataset, val_dataset, name, n=3)

                        print ('---------------------------------------------------')
                        print (f'Average results for {name} dataset')
                        print (avg_dict)
                        print ('---------------------------------------------------')
                    #except:
                        #print (f'Error in {name}')
                        #continue