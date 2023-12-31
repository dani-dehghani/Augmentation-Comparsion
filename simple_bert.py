import pandas as pd
import numpy as np
import shutil
import glob
from simpletransformers.classification import ClassificationModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(100)

class SimpleBert:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # self.args = {
        # 'output_dir': f'./models/bert/{self.dataset_name}',
        # 'reprocess_input_data': True,
        # 'overwrite_output_dir': True,
        # 'train_batch_size': 8,
        # 'num_train_epochs': 10,
        # 'use_multiprocessing': False,
        # 'use_multiprocessing_for_evaluation': False,
        # }
        self.args = {
        'output_dir': f'./models/bert/{self.dataset_name}',
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'train_batch_size': 16,
        'num_train_epochs': 4,
        'learning_rate': 2e-5,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'gradient_accumulation_steps': 2,
        'max_seq_length': 128,
        }
        self.train = None
        self.test = None
        self.model = None
        self.result = None
        self.num_labels = None
        self.result_metrics = None

    def load_data(self):
        encoder = LabelEncoder()
        self.train = pd.read_csv(f'data/original/{self.dataset_name}/train.csv').sample(frac=0.1) # shuffle
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
        self.model.train_model(self.train)

    def compute_metrics(self, preds, labels):
        self.result_metrics
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='macro')
        rec = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')

        return {
            'acc': acc,
            'f1': f1,
            'prec': prec,
            'rec': rec
        }
    
    def evaluate_model(self):
        self.result, _, _ = self.model.eval_model(self.test, compute_metrics=self.compute_metrics)
        self.result_metrics = self.result
        return self.result_metrics

    # def save_model(self, output_dir):
    #     self.model.save_model(output_dir)
    # run model n=3 times and save the best model and average metrics
    def run_n_times(self, n):
        avg_dict = {}
        for i in range(n):
            self.model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=self.num_labels, cuda_device=0, use_cuda=True, args=self.args)
            self.train_model()
            self.evaluate_model()
            for key, value in self.result_metrics.items():
                if key in avg_dict:
                    avg_dict[key] += value
                else:
                    avg_dict[key] = value
                    
        # Calculate the average values of the metrics
        avg_dict = {key: value / n for key, value in avg_dict.items()}
        self.result_metrics = avg_dict
        return avg_dict

    def save_results(self, output_file):
        with open(output_file, 'w') as f:
            if 'compute_metrics' in self.result_metrics:
                metrics = self.result_metrics['compute_metrics']
                for key, value in metrics.items():
                    f.write(f"{key}: {round(value,4)}\n")
            for key, value in self.result_metrics.items():
                if key != 'compute_metrics':
                    f.write(f"{key}: {round(value,4)}\n")

    def clean_up(self):
        #remove all checkpoints folders 
        checkpoints_folder = f'./models/bert/{self.dataset_name}'
        pattern = f'{checkpoints_folder}/checkpoint-*'

        for folder in glob.glob(pattern):
            shutil.rmtree(folder)

    def extract_pre_last_layer(self, text):
        # Tokenize the input text
        tokenizer = self.model.tokenizer
        inputs = tokenizer(text, return_tensors="pt")

        # Move input tensors to the same device as the model (CPU or GPU)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get the output from the base model (layer before the last layer)
        with torch.no_grad():
            base_model_output = self.model.model.distilbert(**inputs)

        hidden_states = base_model_output.last_hidden_state
        return hidden_states
