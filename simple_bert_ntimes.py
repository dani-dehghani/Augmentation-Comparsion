import pandas as pd
import numpy as np
import shutil
import glob
from simpletransformers.classification import ClassificationModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import wandb
from wandb.keras import WandbCallback


wandb.login()


np.random.seed(100)

class SimpleBert:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.args = {
        'output_dir': f'./models/bert/full/{self.dataset_name}',
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'train_batch_size': 8,
        'num_train_epochs': 2,
        #'learning_rate': 2e-5,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        #'weight_decay': 0.01,
        #'warmup_steps': 500,
        'gradient_accumulation_steps': 2,        
        }
        self.train = None
        self.test = None
        self.model = None
        self.result = None
        self.num_labels = None
        self.result_metrics = None
        self.n_times_results = None
        self.callbacks = []

    def load_data(self):
        encoder = LabelEncoder()
        self.train = pd.read_csv(f'data/original/{self.dataset_name}/train.csv').sample(frac=1) # shuffle # change fraction of the dataset here
        self.test_embedding = pd.read_csv(f'data/original/{self.dataset_name}/test.csv') # this will be used only for embedding
        self.test = pd.read_csv(f'data/original/{self.dataset_name}/test.csv')
        self.train = self.train[['text', 'class']]
        self.test = self.test[['text', 'class']]
        self.train.columns = ['text', 'labels']
        self.test.columns = ['text', 'labels']
        self.train['labels'] = encoder.fit_transform(self.train['labels']) # encode the labels to start from 0
        self.test['labels'] = encoder.transform(self.test['labels'])
        self.num_labels = self.train['labels'].nunique()

    def train_model(self):        
        self.model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=self.num_labels, cuda_device=0, use_cuda=True, args=self.args)

        callbacks = []

        # Add the W&B callback
        wandb_callback = WandbCallback(
            monitor='val_loss',
            mode='auto',
            save_model=True,
            verbose=0
        )
        callbacks.append(wandb_callback)

        # Add any other callbacks you want to use here

        self.model.train_model(self.train, callbacks=callbacks)  


    def compute_metrics(self, preds, labels):        
        if self.num_labels == 2:
            average = 'binary'
        else:
            average = 'macro'
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average=average)
        rec = recall_score(labels, preds, average=average)
        f1 = f1_score(labels, preds, average=average)

        return {
            'acc': acc,
            'f1': f1,
            'prec': prec,
            'rec': rec
        }
    
    def evaluate_model(self):
        self.result, _, _ = self.model.eval_model(self.test, compute_metrics=self.compute_metrics)
        self.result_metrics = self.result
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Log metrics to W&B
        for metric_name, metric_value in self.result_metrics.items():
            wandb.log({metric_name: metric_value})

        return self.result_metrics
        


    def run_n_times(self, n):
        dict_list = []                
        for i in range(n):

            wandb.init(
                
                project="Aug",
                config={
                "Ite": i,
                "architecture": "Transformer",
                "dataset name": self.dataset_name,
                "dataset type": 'Original',
                "dataset percentage": 'Full dataset',
                "dataset number of example": None
                
                }         
            )

            self.model = None            
            self.model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=self.num_labels, cuda_device=0, use_cuda=True, args=self.args)
            self.train_model()
            self.evaluate_model()
            metrics = self.save_results(write_to_file=False)

            self.extract_pre_last_layer(self.test_embedding, self.dataset_name)


            temp_dict = {}
            for key, value in metrics.items():         
                temp_dict[key] = value
            dict_list.append(temp_dict)
            # Finish the run
            wandb.finish()

        new_dict = {}
        for i in range(n):
            for key, value in dict_list[i].items():
                if key in new_dict:
                    new_dict[key] += value
                else:
                    new_dict[key] = value
                    
        new_dict = {key: round(value/n,4) for key, value in new_dict.items()}

        self.n_times_results = new_dict
        return new_dict


    def save_results(self, output_file=None, write_to_file=False, n_times=False):

        def format_metrics(metrics):
            return "\n".join(f"{key}: {round(value, 4)}" for key, value in metrics.items())

        if n_times:
            metrics = self.n_times_results
        elif 'compute_metrics' in self.result_metrics:
            metrics = self.result_metrics['compute_metrics']
        else:
            metrics = self.result_metrics

        result_str = format_metrics(metrics)

        if write_to_file:
            with open(output_file, 'w') as f:
                f.write(result_str)
        round_metrics = {key: round(value, 4) for key, value in metrics.items()}
        return round_metrics



    def clean_up(self):
        #remove all checkpoints folders 
        checkpoints_folder = f'./models/bert/full/{self.dataset_name}'
        pattern = f'{checkpoints_folder}/checkpoint-*'

        for folder in glob.glob(pattern):
            shutil.rmtree(folder)

    def extract_pre_last_layer(self, text, dataset_name):
        text_list = text['text'].tolist()
        # Tokenize the input text
        tokenizer = self.model.tokenizer
        inputs = tokenizer(text_list, return_tensors="pt")

        # Move input tensors to the same device as the model (CPU or GPU)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get the output from the base model (layer before the last layer)
        with torch.no_grad():
            base_model_output = self.model.model.distilbert(**inputs)

        hidden_states = base_model_output.last_hidden_state
        cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
        os.makedirs("embeddings/original/bert", exist_ok=True)
        save_path = f'embeddings/original/bert/{dataset_name}.npy'

        # Save the embeddings as a NumPy .npy file
        np.save(save_path, cls_embeddings.cpu().numpy())



    def saving_embeddings(self, test_dataset, dataset_name):
        embeddings = []
        with torch.no_grad():
            for x, _ in test_dataset:

                # Tokenize the input text
                tokenizer = self.model.tokenizer
                inputs = tokenizer(x, return_tensors="pt")

                # Move input tensors to the same device as the model (CPU or GPU)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Get the output from the base model (layer before the last layer)
                with torch.no_grad():
                    base_model_output = self.model.model.distilbert(**inputs)

                hidden_states = base_model_output.last_hidden_state

                # Extract embeddings of [CLS] tokens
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # Assuming you want CPU numpy arrays

        embeddings = torch.cat(cls_embeddings, dim=0)

        os.makedirs("embeddings/original/bert", exist_ok=True)
        save_path = f'embeddings/original/bert/{dataset_name}.npy'

        # Save the embeddings as a NumPy .npy file
        np.save(save_path, embeddings.cpu().numpy())

    