import tensorflow as tf
from config import config
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
import math
import time

class PropensityDropout(tf.keras.Model):
    # with propensity-dropout
    def __init__(self):
        super().__init__('PropensityDropout')
        
        input_layer = Input(shape=(config.number_of_covariates_transformed, ))
        # input_layer = Input(batch_shape=(config.number_of_covariates_transformed/7, config.number_of_covariates_transformed))
        
        # Counterfactual Network
        shared_layer_1 = Dense(200, activation='relu', name='shared_layer_1')(input_layer)
        shared_layer_1_dropout = Dropout(rate=0, name = 'shared_layer_1_dropout')(shared_layer_1)
        shared_layer_2 = Dense(200, activation='relu', name='shared_layer_2')(shared_layer_1_dropout)
        shared_layer_2_dropout = Dropout(rate=0, name = 'shared_layer_2_dropout')(shared_layer_2)
        
        treated_layer_1 = Dense(200, activation='relu', name='treated_layer_1')(shared_layer_2_dropout)
        treated_layer_1_dropout = Dropout(rate=0, name = 'treated_layer_1_dropout')(treated_layer_1)
        treated_output= Dense(1)(treated_layer_1_dropout)
        
        control_layer_1 = Dense(200, activation='relu', name='control_layer_1')(shared_layer_2_dropout)
        control_layer_1_dropout = Dropout(rate=0, name = 'control_layer_1_dropout')(control_layer_1)
        control_output= Dense(1)(control_layer_1_dropout)
        
        self.treated_model = Model(input_layer, treated_output)
        self.control_model = Model(input_layer, control_output)
        
        self.treated_model.compile(optimizer = Adam(lr=0.01),
                              loss = mean_squared_error,
                              metrics=['acc', 'mse'])
        
        self.control_model.compile(optimizer = Adam(lr=0.01),
                              loss = mean_squared_error,
                              metrics=['acc', 'mse'])
        
    def dropout_rate(self, p):
        # print('p', p)
        en = -(p * math.log(p) + (1-p) * math.log(1-p))
        rate = 1 - 1/2 - 1/2 * en
        return rate
        
    def train_model(self, X, P, Y, epoch, units):
        loss_collection = []
        if epoch%2 == 0:
            start = time.time()
            for unit in range(units):
                drate = self.dropout_rate(P[unit])
                self.treated_model.get_layer("shared_layer_1_dropout").rate = drate
                self.treated_model.get_layer("shared_layer_2_dropout").rate = drate
                self.treated_model.get_layer("treated_layer_1_dropout").rate = drate
                # print('shape of np.expand_dims(X[unit]',np.expand_dims(X[unit], axis = 0).shape)
                # print('shape of X[unit]', X[unit].shape)
                pred = self.treated_model.predict(np.expand_dims(X[unit], axis = 0))
                # pred = self.treated_model(X[unit])
                # print('shape of pred', pred.shape)
                loss = self.treated_model.loss(np.expand_dims(Y[unit], axis = 0), pred)
                # pred = self.treated_model(X[unit])
                # loss = self.treated_model.loss(Y[unit])
                loss_collection.append(loss)
            end = time.time()
                
            f = open('time_for_propensity_drop', 'a')
            f.writelines(str(end- start) + '\n')
            f.close()
            total_loss = tf.reduce_mean(loss_collection)
            grads = tf.gradients(total_loss, self.treated_model.trainable_variables)
            self.treated_model.optimizer.apply_gradients(zip(grads, self.treated_model.trainable_variables))
        else:
            for unit in range(units):
                drate = self.dropout_rate(P[unit])
                self.control_model.get_layer("shared_layer_1_dropout").rate = drate
                self.control_model.get_layer("shared_layer_2_dropout").rate = drate
                self.control_model.get_layer("control_layer_1").rate = drate
                
                pred = self.control_model.predict(np.expand_dims(X[unit], axis = 0))
                pred = self.control_model(X[unit])
                print('shape of pred', pred.shape)
                loss = self.control_model.loss(np.expand_dims(Y[unit], axis = 0), pred)
                # pred = self.control_model(X[unit])
                # loss = self.control_model.loss(Y[unit])
                loss_collection.append(loss)
            total_loss = tf.reduce_mean(loss_collection)
            grads = tf.gradients(total_loss, self.control_model.trainable_variables)
            self.control_model.optimizer.apply_gradients(zip(grads, self.control_model.trainable_variables))
            
        return total_loss
    
class PropensityDropout2(tf.keras.Model):
    # with fixed dropout rate
    def __init__(self):
        super().__init__('PropensityDropout')
        
        input_layer = Input(shape=(config.number_of_covariates_transformed,))
        
        # Propensity Network
        p_layer_1 = Dense(25, activation='relu', name='p_layer_1')(input_layer)
        p_layer_2 = Dense(25, activation='relu', name='p_layer_2')(p_layer_1)
        p_layer_3 = Dense(1, name='p_layer_3')(p_layer_2)
        
        self.propensity_model = Model(input_layer, p_layer_3)
        
        self.propensity_model.compile(optimizer = Adam(lr=0.1),
                                 loss = 'binary_crossentropy',
                                 metrics=['accuracy'])
        
        # Counterfactual Network
        shared_layer_1 = BatchNormalization()(Dense(200, activation='relu', name='shared_layer_1')(input_layer))
        shared_layer_1_dropout = Dropout(rate=0.2, name = 'shared_layer_1_dropout')(shared_layer_1)
        shared_layer_2 = BatchNormalization()(Dense(200, activation='relu', name='shared_layer_2')(shared_layer_1_dropout))
        shared_layer_2_dropout = Dropout(rate=0.2, name = 'shared_layer_2_dropout')(shared_layer_2)
        
        treated_layer_1 = BatchNormalization()(Dense(200, activation='relu', name='treated_layer_1')(shared_layer_2_dropout))
        treated_layer_1_dropout = Dropout(rate=0.2, name = 'treated_layer_1_dropout')(treated_layer_1)
        treated_output= Dense(1)(treated_layer_1_dropout)
        
        control_layer_1 = BatchNormalization()(Dense(200, activation='relu', name='control_layer_1')(shared_layer_2_dropout))
        control_layer_1_dropout = Dropout(rate=0.2, name = 'control_layer_1_dropout')(control_layer_1)
        control_output= Dense(1)(control_layer_1_dropout)
        
        self.treated_model = Model(input_layer, treated_output)
        self.control_model = Model(input_layer, control_output)
        
        self.treated_model.compile(optimizer = Adam(lr=0.1),
                              loss = mean_squared_error,
                              metrics=['acc', 'mse'])
        
        self.control_model.compile(optimizer = Adam(lr=0.1),
                              loss = mean_squared_error,
                              metrics=['acc', 'mse'])
        
    def train_model(self, X, Y, epoch, propensity_scores):
        if epoch%2 == 0:   
            pred = self.treated_model(X)
            # print('shape of pred', pred)
            # print('shape of Y', Y.shape)
            # print('shape of P', P.shape)
            loss = tf.reduce_mean(tf.math.square((pred - Y))/propensity_scores)
            # print('loss', loss)
            grads = tf.gradients(loss, self.treated_model.trainable_variables)
            self.treated_model.optimizer.apply_gradients(zip(grads, self.treated_model.trainable_variables))
            return loss
        else:
            pred = self.control_model(X)
            # loss = tf.reduce_mean(tf.math.square((pred - Y))/P)
            loss = tf.reduce_mean(tf.math.square((pred - Y))/propensity_scores)
            # print('loss', loss)
            grads = tf.gradients(loss, self.control_model.trainable_variables)
            self.control_model.optimizer.apply_gradients(zip(grads, self.control_model.trainable_variables))
            return loss

    
class AutoEncoder(tf.keras.Model):
    # learning representations
    def __init__(self):
        super().__init__('AutoEncoder')
        # this is our input placeholder
        input_img = Input(batch_shape=(config.number_of_units, config.number_of_covariates_transformed))
        
        encoded1 = Dense(50, 
                         activation='relu',
                         # kernel_regularizer=regularizers.l2(0.01),
                          # activity_regularizer=regularizers.l1(0.01),
                         )(input_img)
        encoded2 = BatchNormalization()(encoded1)
        encoded3 = Dense(25, 
                         # activation='relu',
                         # kernel_regularizer=regularizers.l2(0.01),
                         # activity_regularizer=regularizers.l1(0.01),
                         )(encoded2)
        # encoded3 = Dense(78, 
        #                  activation='relu', 
        #                  # activity_regularizer=regularizers.l1(0.01),
        #                  )(encoded2)
        
        decoded1 = Dense(50, 
                         activation='relu',
                         # kernel_regularizer=regularizers.l2(0.01),
                          # activity_regularizer=regularizers.l1(0.01)
                         )(encoded1)
        decoded2 = BatchNormalization()(decoded1)
        decoded3 = Dense(78, 
                         # activation='relu',
                         # kernel_regularizer=regularizers.l2(0.01),
                          # activity_regularizer=regularizers.l1(0.01)
                         )(decoded2)
        # decoded3 = Dense(78, activation='relu')(decoded2)
        
        self.decoder = Model(input_img, decoded3)
        self.encoder = Model(input_img, encoded3)
        # self.inf_loss_func = tf.nn.l2_loss
        self.optimizer = tf.compat.v1.train.AdamOptimizer()
        # self.decoder.compile(optimizer=self.optimizer, loss=self.inf_loss_func)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.decoder(x)
        return x
    
    def call_encoder(self, inputs):
        return self.encoder(inputs)
    
    def cross_entropy(self, x, y, axis=-1):
        safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
        return -tf.reduce_sum(x * tf.log(safe_y), axis)

    def train_model(self, X, phi_propensity_scores, T_dist_array):
        with tf.GradientTape() as tape:     
            # print('X', X)
            psi = self(X)
            inf_loss = tf.math.reduce_mean(
                tf.math.sqrt(
                    tf.math.reduce_sum(
                        tf.math.square(
                            tf.math.subtract(psi, X)), axis = -1)))
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = T_dist_array, logits= phi_propensity_scores))
            # print('inf_loss', tf.keras.backend.eval(inf_loss))
            print('ce_loss', tf.keras.backend.eval(ce_loss))
            loss = config.lambda_2 * inf_loss + config.lambda_1 * tf.cast(ce_loss, tf.float32)
        grads = tf.gradients(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))
        return loss


