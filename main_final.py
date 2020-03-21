import numpy as np
from config import config as config
import logging
import sys
import pandas as pd
from features import load_data
from sklearn.linear_model import LogisticRegression, Lasso
import os
import tensorflow.keras.backend as K
import tensorflow as tf
from models import PropensityDropout2
import random
from sklearn.preprocessing import MinMaxScaler
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.enable_eager_execution()
        
if __name__ == '__main__':
    with tf.Session() as sess:    
        df_covariates = pd.read_csv(config.covariates_file)
    
        X = load_data(data=df_covariates, features=config.feature_cols)
        scaler = MinMaxScaler()
        
        for i in range(4, 5):
            writer = tf.summary.FileWriter('logs_final/counterfactual_models/dataset'+str(i) +'/',sess.graph)
            last_batch = 0
            
            network = PropensityDropout2()
            # network.treated_model.load_weights(config.counterfactual_model_dir+str(i) + '/' + 'treated' +'/')
            # network.control_model.load_weights(config.counterfactual_model_dir+str(i) + '/' + 'control' +'/')
            for epoch in range(config.train_epochs):
                # lr = LogisticRegression()
                outcome_files = os.listdir(config.outcome_file_dir[i])
                random.shuffle(outcome_files)
                epoch_loss = 0.0
                # epoch_acc = 0.0
                
                if epoch%2 == 0:
                    for j in range(config.train_batches): 
                        outcome_file = outcome_files[j] # each file is a batch
                        lasso = Lasso()
                        Y_true = []
                        df_outcomes = pd.read_csv(os.path.join(config.outcome_file_dir[i], outcome_file))
                        treated_group = df_outcomes.loc[df_outcomes['z'] == 1]
                        treated_group_index = df_outcomes[df_outcomes['z'] == 1].index.tolist()
                        
                        # apply lasso to eliminate unhelpful covariates
                        T = df_outcomes['z'].values
                        Y0 = df_outcomes['y0'].values
                        Y1 = df_outcomes['y1'].values
                        # print('Y1', type(Y1))
                        for individual in range(0, 4802):
                            if T[individual] == 0:
                                Y_true.append(Y0[individual])
                            else:
                                Y_true.append(Y1[individual])   
                        lasso.fit(X, np.array(Y_true))
                        coef = lasso.coef_
                        coef[coef != 0] = 1
                        coef[coef != 1] = 0
                        X_samples = np.multiply(X, coef)
                        
                        # scale inputs into the range (0, 1)
                        scaler.fit(X_samples)
                        X_samples = scaler.transform(X_samples)
                        X_samples = np.float32(X_samples)
                        
                        # use logistic regression to estimate propensity scores
                        # lr.fit(X_samples, T)
                        # propensity_scores = lr.predict_proba(X_samples)[:, 1]
                        
                        # print('shape of propensity_scores', propensity_scores[treated_group_index].shape)
                        # print('shape of X_samples', X_samples[treated_group_index, :].shape)
                        # print('shape of treated_group', treated_group['y1'].values.shape)
                        # print('\n')
                        # loss = network.train_model(X_samples[treated_group_index, :], 
                        #                             # np.expand_dims(propensity_scores[treated_group_index], axis = -1), 
                        #                             np.expand_dims(treated_group['y1'].values, axis = -1), 
                        #                             epoch, 
                        #                             propensity_scores[treated_group_index])
                        
                        
                        
                        # loss = tf.keras.backend.eval(loss)
                        history = network.treated_model.fit(X_samples[treated_group_index, :], 
                                                          np.expand_dims(treated_group['y1'].values, axis = -1),
                                                          batch_size = len(treated_group_index))
                        # print('loss', loss)
                        # epoch_loss += loss
                        epoch_loss += history.history['loss'][0]
                    
                        # batch_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_loss", simple_value=loss)])
                        # writer.add_summary(batch_loss_summary, j+last_batch)
                        batch_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_loss", simple_value=history.history['loss'][0])])
                        writer.add_summary(batch_loss_summary, j+last_batch)
                        
                else:
                    for j in range(config.train_batches): 
                        outcome_file = outcome_files[j] # each file is a batch
                        lasso = Lasso()
                        Y_true = []
                        df_outcomes = pd.read_csv(os.path.join(config.outcome_file_dir[i], outcome_file))
                        control_group = df_outcomes.loc[df_outcomes['z'] == 0]
                        control_group_index = df_outcomes[df_outcomes['z'] == 0].index.tolist()
                        
                        # apply lasso to eliminate unhelpful covariates
                        T = df_outcomes['z'].values
                        Y0 = df_outcomes['y0'].values
                        Y1 = df_outcomes['y1'].values
                        # print('Y1', type(Y1))
                        for individual in range(0, 4802):
                            if T[individual] == 0:
                                Y_true.append(Y0[individual])
                            else:
                                Y_true.append(Y1[individual])   
                        lasso.fit(X, np.array(Y_true))
                        coef = lasso.coef_
                        coef[coef != 0] = 1
                        coef[coef != 1] = 0
                        X_samples = np.multiply(X, coef)
                        
                        # scale inputs into the range (0, 1)
                        scaler.fit(X_samples)
                        X_samples = scaler.transform(X_samples)
                        X_samples = np.float32(X_samples)
                        
                        # use logistic regression to estimate propensity scores
                        # lr.fit(X_samples, T)
                        # propensity_scores = lr.predict_proba(X_samples)[:, 0]
                        

                        # loss = network.train_model(X_samples[control_group_index, :], 
                        #                             # np.expand_dims(propensity_scores[control_group_index], axis = -1), 
                        #                             np.expand_dims(control_group['y0'].values, axis = -1), 
                        #                             epoch, 
                        #                             propensity_scores[control_group_index])
                        history = network.treated_model.fit(X_samples[control_group_index, :], 
                                                          np.expand_dims(control_group['y0'].values, axis = -1),
                                                          batch_size = len(control_group_index))
                        
                        # loss = tf.keras.backend.eval(loss)
                        # print('loss', loss)
                        # epoch_loss += loss
                        epoch_loss += history.history['loss'][0]
                    
                        # batch_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_loss", simple_value=loss)])
                        # writer.add_summary(batch_loss_summary, j+last_batch)
                        batch_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_loss", simple_value=history.history['loss'][0])])
                        writer.add_summary(batch_loss_summary, j+last_batch)
                    
                last_batch += config.train_batches
                epoch_summary = tf.Summary(value=[tf.Summary.Value(tag="epoch_loss", simple_value=epoch_loss/config.train_batches)])
                writer.add_summary(epoch_summary, epoch)

            
                network.treated_model.save_weights(config.counterfactual_model_dir+str(i) + '/' + 'treated' +'/')
                network.control_model.save_weights(config.counterfactual_model_dir+str(i) + '/' + 'control' +'/')
                
                        
                
                