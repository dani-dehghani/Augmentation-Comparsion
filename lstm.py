import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import os
import json
import random
import wandb
from wandb.keras import WandbCallback



#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(100)
random.seed(100)
tf.random.set_seed(100)


wandb.login()


class LSTM:
    def __init__(self, dims, w2v_path,aug_method,percentage, num_example,fulldataset= False ,max_seq_len=20, batch_size=128, epochs=20, chunk_size=1000):
        self.aug_method = aug_method
        self.percentage = percentage
        self.num_example = num_example
        self.fulldataset = fulldataset
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.epochs = epochs
        with open(w2v_path, 'rb') as f:
            self.w2v = pickle.load(f)
        self.model = None
        self.n_classes = None
        self.history = None
        self.metrics = None
        self.callbacks = None
        


    def build_lstm(self):
        if self.n_classes > 2:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        else:
            loss = 'binary_crossentropy'
            activation = 'sigmoid'

        input_layer = layers.Input(shape=(self.max_seq_len, 300))
        lstm_1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_layer)
        dropout_rate = 0.5
        dropout_out1 = layers.Dropout(dropout_rate)(lstm_1)
        lstm_2 = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(dropout_out1)
        dropout_out2 = layers.Dropout(dropout_rate)(lstm_2)
        dense_1 = layers.Dense(20, activation='relu')(dropout_out2)
        dense_out = layers.Dense(self.n_classes, activation=activation, kernel_regularizer=regularizers.L2(0.001))(dense_1)
        
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        lstm_model = Model(inputs=input_layer, outputs=dense_out)
        lstm_model.compile(optimizer='adam', loss=loss, metrics=self.metrics)
        #lstm_model.summary()
        self.model = lstm_model

    def prepare_dataset(self, df):
        def generator():
            for _, row in df.iterrows():
                label = row[0]
                sentence = row[1]
                x = np.zeros((self.max_seq_len, 300))
                y = np.zeros(self.n_classes)

                if isinstance(sentence, str):
                    words = sentence.split()[:self.max_seq_len]
                    for k, word in enumerate(words):
                        if word in self.w2v:
                            x[k, :] = self.w2v[word]
                y[label] = 1.0
                yield x, y

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.max_seq_len, 300), dtype=tf.float32),
                tf.TensorSpec(shape=(self.n_classes,), dtype=tf.float32)
            )
        )
        return dataset


    def insert_values(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.n_classes = train_df['class'].nunique()
        unique_classes = train_df['class'].unique()
        labels_map = dict(zip(unique_classes, range(self.n_classes)))

        train_df['class'] = train_df['class'].map(labels_map)
        test_df['class'] = test_df['class'].map(labels_map)

        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=100)
        print(f'Train size: {len(train_df)}\nValidation size: {len(val_df)}\nTest size: {len(test_df)}')

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=100).reset_index(drop=True)

        train_dataset = self.prepare_dataset(train_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = self.prepare_dataset(test_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = self.prepare_dataset(val_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, test_dataset, val_dataset, self.n_classes

    def fit(self, train_dataset, val_dataset):
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        self.build_lstm()
        
        # Create the W&B callback
        wandb_callback = WandbCallback(
            monitor='val_loss',
            mode='auto',
            save_model= True,
            verbose=0
        )
        
        # Add the W&B callback to the list of callbacks
        self.callbacks.append(wandb_callback)
        
        # Train the model with callbacks
        self.history = self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset, callbacks=self.callbacks, verbose=0)

        return self.history

        
    
    def evaluate(self, test_dataset):
        evaluation_metrics = self.model.evaluate(test_dataset, return_dict=True)

        return evaluation_metrics


    def run_n_times(self, train_dataset, test_dataset, val_dataset, dataset_name, n=3):
        dataset_list = ['agnews', 'subj', 'pc', 'yelp', 'cr', 'kaggle_med', 'cardio', 'bbc', 'sst2']
        log_dir = f"logs/fit/lstm/{dataset_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        decay_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001 ,min_lr=0.00001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        self.callbacks = [tensorboard_callback, decay_rate, early_stopping]

        hist_dict = {}
        res_dict = {}
        best_val_loss = float('inf')
        for i in range(n):


            wandb.init(
                
                project="Aug",
                config={
                
                "Ite": i,
                "architecture": "LSTM",
                "dataset name": dataset_name,
                "dataset type": 'Augmented',
                "dataset percentage": self.percentage,
                "aug method": f'{self.aug_method}',
                "examples": self.num_example
                
                }         
            )


            print(f'Run {i+1} of {n}')
            try:
                self.fit(train_dataset, val_dataset)  # Updated to use train_dataset and val_dataset
            except tf.errors.ResourceExhaustedError:
                K.clear_session()
                self.model = None
                self.build_lstm()
                continue
            
            
            res = self.evaluate(test_dataset)  # Updated to use test_dataset
            for metric_name, metric_value in res.items():
                wandb.log({metric_name: metric_value})
            wandb.finish()
            res_dict[i+1] = res
            if self.history.history['val_loss'][-1] < best_val_loss:
                best_val_loss = self.history.history['val_loss'][-1]
                self.model.save(f"models/lstm/full/{dataset_name}_best_model.h5")
                if self.fulldataset == True:
                    self.saving_embeddings(test_dataset, dataset_name)
            self.model.set_weights([np.zeros(w.shape) for w in self.model.get_weights()])

        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4) for metric in res_dict[1].keys()}

        # Save the average results to disk
        #os.makedirs("results/augmented/lstm", exist_ok=True)
        #with open(f"results/augmented/lstm/{self.percentage}/{dataset_name}_{self.percentage}_example_{self.num_example}.txt", "w") as f:
            #for key, value in avg_dict.items():
                #f.write(f"{key}: {value}\n")

        K.clear_session()

        return hist_dict, res_dict, avg_dict

    def extract_pre_last_layer(self, numeric_data):
        # Find the pre-last layer dynamically
        pre_last_layer_index = -2  # Index of the pre-last layer (dense_1 is typically the pre-last layer)
        pre_last_layer = self.model.layers[pre_last_layer_index]

        # Create a model that extracts features from the pre-last layer
        intermediate_layer_model = Model(inputs=self.model.input, outputs=pre_last_layer.output)

        # Extract pre-last layer features from numeric_data
        pre_last_layer_features = intermediate_layer_model.predict(numeric_data)

        return pre_last_layer_features

    def saving_embeddings(self, test_dataset, dataset_name):
        embeddings = []
        with tf.device('/GPU:0'):
            for x, _ in test_dataset:
                pre_last_layer_features = self.extract_pre_last_layer(x)
                embeddings.append(pre_last_layer_features)

        embeddings = tf.concat(embeddings, axis=0)

        # Assertions to check the shape of embeddings
        #assert normalized_embeddings.shape[0] == len(test_dataset) and len(normalized_embeddings.shape) == 2
        os.makedirs("embeddings/original/lstm", exist_ok=True)
        save_path = f'embeddings/original/lstm/{dataset_name}.npy'  # Save as a .npy file

        # Save the embeddings as a NumPy .npy file
        np.save(save_path, embeddings)
