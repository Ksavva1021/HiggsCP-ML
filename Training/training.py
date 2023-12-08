import os
import sys
directory = os.getcwd()
sys.path.append(directory+'/python/')
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from EmbeddedMapping import EmbeddedMapping
import pickle
from lbn import LBN, LBNLayer
import wandb

def main(dataset = 'dataset',
	 inputs=[
         "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1",
         "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2",
         "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1",
         "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
         "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1",
         "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
         "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1",
         "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2",
         "ip_x_1", "ip_y_1", "ip_z_1","ipcov00_1","ipcov01_1","ipcov02_1","ipcov10_1","ipcov11_1","ipcov12_1","ipcov20_1","ipcov21_1","ipcov22_1",
         "ip_x_2", "ip_y_2", "ip_z_2","ipcov00_2","ipcov01_2","ipcov02_2","ipcov10_2","ipcov11_2","ipcov12_2","ipcov20_2","ipcov21_2","ipcov22_2",
         "sv_x_1", "sv_y_1", "sv_z_1","svcov00_1","svcov01_1","svcov02_1","svcov10_1","svcov11_1","svcov12_1","svcov20_1","svcov21_1","svcov22_1",
         "sv_x_2", "sv_y_2", "sv_z_2","svcov00_2","svcov01_2","svcov02_2","svcov10_2","svcov11_2","svcov12_2","svcov20_2","svcov21_2","svcov22_2",
         "pt_1", "pt_2", "pt_vis",
         "met", "metx", "mety",
         ], 
         categorical_inputs=[
         "tau_decay_mode_1","tau_decay_mode_2"
         ],
         targets_1=[
	 "..."
         ],
         targets_2=[
	 "..."
         ],         
         cpweights=[
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm"
         ],
         input_scaling=True,
         output_scaling=True,
         train_test_split=[4,0],
         batch_size = 512*25,
         units_1=((150,)*10),
         units_2=((150,)*5),
         activation='relu',
         kernel_initializer='he_uniform',
         initial_learning_rate=0.01,
         decay_rate = 0.96,
         epochs = 500,
         early_stopping_patience=20,
         save_directory = '',
         model_name='model',
         output_name='info', 
         ):

   print("Inputs for training:",inputs)
   
   # Initialising wandb for Model Loss tracking
   wandb.init(
       # set the wandb project where this run will be logged
       project='Neutrino Regression',
       # track hyperparameters and run metadata
       config={
       "learning_rate": initial_learning_rate,
       "architecture": "DNN",
       "dataset": "HTT",
       "epochs": epochs,
       }
   )
   
   # Shuffling the dataset & setting a mask for train/test splitting
   random_seed = 42
   df = pd.read_pickle("{}/samples/pickle/{}.pkl".format(directory,dataset))
   df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
   df['row_index'] = df.index
   train_mask = (df["row_index"] % train_test_split[0]) != train_test_split[1]

   # setting up separate dataframe for inputs, categorical features, targets, and spinner weights
   df = df.astype(np.float32) 
   df_inputs = df.filter(inputs)
   df_cat = df.filter(categorical_inputs)
   df_targets_1 = df.filter(targets_1)
   df_targets_2 = df.filter(targets_2)   
   df_cpweights = df.filter(cpweights)
   
   # applying the training mask for train/test splitting
   # Train 
   inputs_train = df_inputs[train_mask]
   cat_train = df_cat[train_mask]
   targets_1_train = df_targets_1[train_mask]
   targets_2_train = df_targets_2[train_mask]   
   cpweights_train = df_cpweights[train_mask]
   # Test
   inputs_test  = df_inputs[~train_mask]
   cat_test  = df_cat[~train_mask]
   targets_1_test  = df_targets_1[~train_mask]
   targets_2_test  = df_targets_2[~train_mask]   
   cpweights_test  = df_cpweights[~train_mask]
  
   # Resetting the row indexes for easier manipulation 
   inputs_train.reset_index(drop=True,inplace = True)
   inputs_test.reset_index(drop=True,inplace = True)
   cat_train.reset_index(drop=True,inplace = True)
   cat_test.reset_index(drop=True,inplace = True)
   targets_1_train.reset_index(drop=True,inplace = True)
   targets_1_test.reset_index(drop=True,inplace = True)
   targets_2_train.reset_index(drop=True,inplace = True)
   targets_2_test.reset_index(drop=True,inplace = True)
   
   # making a copy of the inputs before passing them through standardisation 
   raw_inputs_train = inputs_train.copy()
   raw_inputs_test = inputs_test.copy()
   
   # computing mean and variance to standardise the inputs during training
   if input_scaling:
      input_means = inputs_train.mean()
      input_variances   = inputs_train.var()
      input_means_tensor = tf.constant(input_means, dtype=tf.float32)
      input_variances_tensor = tf.constant(input_variances, dtype=tf.float32)
   
   # standardising the target variables
   if output_scaling:
      target_1_means = targets_1_train.mean()
      target_1_stds   = targets_1_train.std()
      targets_1_train = (targets_1_train - target_1_means) / target_1_stds
      targets_1_test = (targets_1_test - target_1_means) / target_1_stds

      target_2_means = targets_2_train.mean()
      target_2_stds   = targets_2_train.std()
      targets_2_train = (targets_2_train - target_2_means) / target_2_stds
      targets_2_test = (targets_2_test - target_2_means) / target_2_stds      

   # setting up embedded mapping due to the presence of categorical features
   print("Lengnth of cat",len(categorical_inputs))
   if len(categorical_inputs) > 0:
      embedded_mappings = {}
      for feature in categorical_inputs:
         embedded_mappings[feature] = EmbeddedMapping(cat_train[feature])
      for feature in categorical_inputs:
         feature_name = '{}_mapping'.format(feature)
         cat_train = cat_train.assign(**{feature_name: cat_train[feature].apply(embedded_mappings[feature].get_mapping)})
         cat_test = cat_test.assign(**{feature_name: cat_test[feature].apply(embedded_mappings[feature].get_mapping)})

   # -------------Training Setup-------------
   # Initialising the model
   model = create_model(len(inputs),
                        len(targets_1),
                        len(targets_2),
                        categorical_inputs,
                        embedded_mappings,
                        units_1,
                        units_2,
                        activation,
                        kernel_initializer,
        		input_scaling,
        		input_means_tensor,
        		input_variances_tensor)
   model.summary()
   
   # Learning Rate, Optimiser, Loss Function
   learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, trainable=False)
   optimizer = tf.keras.optimizers.Nadam(learning_rate)      
   
   loss_dict = create_lossFunction_Neutrino(maeLoss=True,crossProduct=True)
   loss_2_dict = create_lossFunction_PV(maeLoss=True,crossProduct=True)
   loss_names = [key for key in loss_dict]
   for key in loss_2_dict: loss_names.append(key)
   loss_names.append("mean")
   
   loss_names = list(set(loss_names))

   # setting up datasets for training and testing 
   train_dataset = tf_dataset(inputs_train.to_numpy(),targets_1_train.to_numpy(),targets_2_train.to_numpy(),batch_size,True,cat_train['tau_decay_mode_1_mapping'].to_numpy(),cat_train['tau_decay_mode_2_mapping'].to_numpy(), raw_inputs_train.to_numpy())
   test_dataset = tf_dataset(inputs_test.to_numpy(),targets_1_test.to_numpy(),targets_2_test.to_numpy(),batch_size,False,cat_test['tau_decay_mode_1_mapping'].to_numpy(),cat_test['tau_decay_mode_2_mapping'].to_numpy(), raw_inputs_test.to_numpy())
   # ----------------------------------------
 
   # -------------Train/Test Loops-------------
   # Initialising losses, weight, epoch tracking 
   wait = 0
   optimal_loss = float('inf')
   optimal_weights = None
   train_losses = {}
   test_losses = {}
   
   # Training/Test Loops
   for epoch in range(epochs):

      # -------------Training Loop Start-------------
      epoch_losses_train = {key: [] for key in loss_names}      
      for step, (input_batch_train, DM1_batch_train, DM2_batch_train, raw_inputs_batch_train, target_batch_train, target_2_batch_train) in enumerate(train_dataset):
         train_losses, train_losses_2, total_train_loss = train_step(model, optimizer,[input_batch_train, DM1_batch_train, DM2_batch_train],target_batch_train, target_2_batch_train, raw_inputs_batch_train, target_1_stds.to_numpy(), target_1_means.to_numpy(), target_2_stds.to_numpy(), target_2_means.to_numpy(), loss_dict, loss_2_dict)
         
         # for each step: update the mean, but also individual elements in the loss function 
         for key in loss_names:
            if key == "mean":
               epoch_losses_train[key].append(total_train_loss)
            elif key in train_losses:
               epoch_losses_train[key].append(train_losses[key])
            elif key in train_losses_2:
               epoch_losses_train[key].append(train_losses_2[key])
            else:
               epoch_losses_train[key].append(0.)
  
      # calculate the mean loss for each epoch
      mean_train_losses = {key: np.mean(epoch_losses_train[key]) for key in loss_names}

      # -------------Test Loop Start-------------
      epoch_losses_test= {key: [] for key in loss_names}      
      for step_val,(input_batch_test, DM1_batch_test, DM2_batch_test, raw_inputs_batch_test, target_batch_test, target_2_batch_test) in enumerate(test_dataset):
         test_losses, test_losses_2,total_test_loss = test_step(model,[input_batch_test, DM1_batch_test, DM2_batch_test],target_batch_test, target_2_batch_test, raw_inputs_batch_test, target_1_stds.to_numpy(), target_1_means.to_numpy(),target_2_stds.to_numpy(), target_2_means.to_numpy(), loss_dict, loss_2_dict)

         for key in loss_names:
            if key == "mean":
               epoch_losses_test[key].append(total_test_loss)
            elif key in test_losses:
               epoch_losses_test[key].append(test_losses[key])
            elif key in test_losses_2:
               epoch_losses_test[key].append(test_losses_2[key])
            else:
               epoch_losses_test[key].append(0.)

      # calculate the mean loss for each epoch
      mean_test_losses = {key: np.mean(epoch_losses_test[key]) for key in loss_names}

      # Setting up a decaying learning rate (changes at the end of every epoch)
      current_lr = learning_rate
      updated_lr = initial_learning_rate*(decay_rate**(epoch+1))
      learning_rate.assign(updated_lr)
 
      # Logging the individual losses for wandb tracking
      log_dict = {}
      for loss_name in loss_names:
         log_dict[f"Train {loss_name}"] = mean_train_losses[loss_name]
         log_dict[f"Test {loss_name}"] = mean_test_losses[loss_name]
      wandb.log(log_dict) 

      # Printing the Epoch Number + Different Losses on the command line while running
      template = 'Epoch {}, ' + ', '.join(['{} Train Loss: {}, {} Test Loss: {}' for _ in loss_names])
      values = [epoch + 1]
      for loss_name in loss_names:
          values.extend([loss_name, mean_train_losses[loss_name], loss_name, mean_test_losses[loss_name]])
      print(template.format(*values))

      # Setting up early stopping
      wait += 1
      if mean_test_losses["mean"] < optimal_loss:
         optimal_loss = mean_test_losses["mean"]
         optimal_weights = model.get_weights()
         wait = 0
      if wait >= early_stopping_patience:
         break         
   
   wandb.finish() # wandb logging ends here      

   # saving the model
   if optimal_weights is not None:
      model.set_weights(optimal_weights)
      model.save('{}/{}'.format(save_directory,model_name)) 

   # making predictions and saving the results in a pkl file
   pred = model.predict([inputs_test, cat_test['tau_decay_mode_1_mapping'], cat_test['tau_decay_mode_2_mapping']])
   info = {
        'inputs' : raw_inputs_test,
        'inputs_norm': inputs_test,
        'cat'    : cat_test,
        'targets_1': targets_1_test,
        'targets_2': targets_2_test,
        'predictions': pred,
        'cpweights': cpweights_test, 
        'target_1_means': target_1_means,
        'target_1_stds' : target_1_stds,
        'target_2_means': target_2_means,
        'target_2_stds' : target_2_stds        
        }
   file_path = '{}/{}.pkl'.format(save_directory,output_name)
   with open(file_path, 'wb') as file:
      pickle.dump(info, file)

# ------------Extra Utilities------------
def tf_dataset(inputs, targets, targets_2, batch_size, shuffle=False, *args):
   nevents = len(inputs)
   data = (inputs,*args,targets,targets_2)
   dataset = tf.data.Dataset.from_tensor_slices(data)
   if shuffle:
      dataset = dataset.shuffle(nevents)
   dataset = dataset.batch(batch_size)
   return dataset

def create_model(input_shape, output_shape_Neutrino, output_shape_PV, categorical_inputs, embedded_mappings, units_1, units_2, activation, kernel_initializer, input_scaling, *args):
   inputs = []
   models = []
   
   # Continuous input layer
   x1 = tf.keras.Input(input_shape, name = 'cont_input')
   inputs.append(x1)

   # Normalisation layer for standardising the inputs
   if input_scaling:
      input_means = args[0]
      input_variances = args[1]
      normalisation_layer = tf.keras.layers.Normalization(mean=input_means, variance=input_variances, name = "norm_layer")
      x = normalisation_layer(x1)
      models.append(x)

   # Categorical input processing
   if len(categorical_inputs) > 0:
      for feature in categorical_inputs:
         input_layer = tf.keras.Input(shape=(1,), dtype='int32', name='{}_input'.format(feature))
         embedding_layer = tf.keras.layers.Embedding(output_dim = 1, input_dim = embedded_mappings.get(feature,None).num_values, input_length=1, name='{}_embedding'.format(feature))(input_layer)
         embedding_layer = tf.keras.layers.Flatten()(embedding_layer)
         inputs.append(input_layer)
         models.append(embedding_layer)
  
   # Concatenate categorical and continuous inputs
   x = tf.keras.layers.Concatenate(name="concat_categorical_plus_continuous")(models)
   
   x1 = x
   # Building the model architecture
   for idx, nodes in enumerate(units_1):

      dense_layer = tf.keras.layers.Dense(nodes, use_bias = True, kernel_initializer = kernel_initializer, name = "dense_layer{}".format(idx+1))
      x1 = dense_layer(x1)

      activation_layer = tf.keras.layers.Activation(activation, name = "act_layer{}".format(idx+1))
      x1 = activation_layer(x1)

      batchnorm_layer = tf.keras.layers.BatchNormalization(dtype="float32", name = "batchnorm_layer{}".format(idx+1))
      x1 = batchnorm_layer(x1)

   # Neutrino Output layer 
   outputs_Neutrino = tf.keras.layers.Dense(output_shape_Neutrino, use_bias=True, kernel_initializer = kernel_initializer, name = "output_Neutrino")(x1)
   
   x2 = tf.keras.layers.Concatenate(name="concat_NNs")([x,outputs_Neutrino])
   # Building the model architecture
   for idx, nodes in enumerate(units_2):

      dense_layer = tf.keras.layers.Dense(nodes, use_bias = True, kernel_initializer = kernel_initializer, name = "2_dense_layer{}".format(idx+1))
      x2 = dense_layer(x2)

      activation_layer = tf.keras.layers.Activation(activation, name = "2_act_layer{}".format(idx+1))
      x2 = activation_layer(x2)

      batchnorm_layer = tf.keras.layers.BatchNormalization(dtype="float32", name = "2_batchnorm_layer{}".format(idx+1))
      x2 = batchnorm_layer(x2)   

   # PV Output layer 
   outputs_PV = tf.keras.layers.Dense(output_shape_PV, use_bias=True, kernel_initializer = kernel_initializer, name = "output_PV")(x2)

   model = tf.keras.Model(inputs = inputs, outputs=[outputs_Neutrino,outputs_PV], name="regression")
   return model
     
def create_lossFunction_Neutrino(maeLoss=False,crossProduct=False):
   # Initialize an empty dictionary to store different loss functions
   losses = {}

   if maeLoss:

      @tf.function
      def mae_Neutrino(**kwargs):
         targets = kwargs["targets"] 
         predictions = kwargs["predictions"] 
         
         # Compute MAE Loss
         loss = tf.reduce_mean(tf.abs(targets-predictions))
         return loss

      losses['mae_Neutrino'] = mae_Neutrino
   
   if crossProduct:

      def crossProduct_Lead(**kwargs):
         targets = kwargs["targets"]
         predictions = kwargs["predictions"]
         raw = kwargs['raw_inputs']
         stdev = kwargs['stdev']
         mean = kwargs['mean']

         # De-standardising the targets & predictions
         targets = (stdev * targets) + mean
         predictions = (stdev * predictions) + mean

         px = raw[:,0]+raw[:,8]+raw[:,16]
         py = raw[:,1]+raw[:,9]+raw[:,17]
         pz = raw[:,2]+raw[:,10]+raw[:,18]
         E  = raw[:,3]+raw[:,11]+raw[:,19]

         # Formula: a x b / |a x b|
         a = tf.stack([px, py, pz], axis=-1)
         b_target = tf.stack([targets[:,0], targets[:,1], targets[:,2]], axis=-1)
         b_prediction = tf.stack([predictions[:,0], predictions[:,1], predictions[:,2]], axis=-1)

         target = tf.linalg.cross(a, b_target)
         target_norm = tf.norm(target,axis=-1)
         target = tf.math.divide(target, target_norm[:, tf.newaxis])

         prediction = tf.linalg.cross(a, b_prediction)
         prediction_norm = tf.norm(prediction,axis=-1)
         prediction = tf.math.divide(prediction, prediction_norm[:, tf.newaxis])
                    
         loss = 0.1*tf.reduce_sum(tf.reduce_mean(tf.abs(target - prediction),axis=0))        
         return loss

      losses['CP_Lead_Neutrino'] = crossProduct_Lead

      def crossProduct_Sublead(**kwargs):
         targets = kwargs["targets"]
         predictions = kwargs["predictions"]
         raw = kwargs['raw_inputs']
         stdev = kwargs['stdev']
         mean = kwargs['mean']

         # De-standardising the targets & predictions
         targets = (stdev * targets) + mean
         predictions = (stdev * predictions) + mean

         px = raw[:,4]+raw[:,12]+raw[:,20]
         py = raw[:,5]+raw[:,13]+raw[:,21]
         pz = raw[:,6]+raw[:,14]+raw[:,22]
         E  = raw[:,7]+raw[:,15]+raw[:,23]

         # Formula: a x b / |a x b|
         a = tf.stack([px, py, pz], axis=-1)
         b_target = tf.stack([targets[:,3], targets[:,4], targets[:,5]], axis=-1)
         b_prediction = tf.stack([predictions[:,3], predictions[:,4], predictions[:,5]], axis=-1)

         target = tf.linalg.cross(a, b_target)
         target_norm = tf.norm(target,axis=-1)
         target = tf.math.divide(target, target_norm[:, tf.newaxis])

         prediction = tf.linalg.cross(a, b_prediction)
         prediction_norm = tf.norm(prediction,axis=-1)
         prediction = tf.math.divide(prediction, prediction_norm[:, tf.newaxis])

         loss = 0.1*tf.reduce_sum(tf.reduce_mean(tf.abs(target - prediction),axis=0))
         return loss

      losses['CP_Sublead_Neutrino'] = crossProduct_Sublead

   return losses

def create_lossFunction_PV(maeLoss=False,crossProduct=False):
   # Initialize an empty dictionary to store different loss functions
   losses = {}

   if maeLoss:

      @tf.function
      def mae_PV(**kwargs):
         targets = kwargs["targets"] 
         predictions = kwargs["predictions"] 
         
         # Compute MAE Loss
         loss = tf.reduce_mean(tf.abs(targets-predictions))
         return loss

      losses['mae_PV'] = mae_PV

   if crossProduct:

      def crossProduct_PV_Lead(**kwargs):
         targets = kwargs["targets"]
         predictions = kwargs["predictions"]
         raw = kwargs['raw_inputs']
         stdev = kwargs['stdev']
         mean = kwargs['mean']

         # De-standardising the targets & predictions
         targets = (stdev * targets) + mean
         predictions = (stdev * predictions) + mean

         px = raw[:,0]+raw[:,8]+raw[:,16]
         py = raw[:,1]+raw[:,9]+raw[:,17]
         pz = raw[:,2]+raw[:,10]+raw[:,18]
         E  = raw[:,3]+raw[:,11]+raw[:,19]

         # Formula: a x b / |a x b|
         a = tf.stack([px, py, pz], axis=-1)
         b_target = tf.stack([targets[:,0], targets[:,1], targets[:,2]], axis=-1)
         b_prediction = tf.stack([predictions[:,0], predictions[:,1], predictions[:,2]], axis=-1)

         target = tf.linalg.cross(a, b_target)
         target_norm = tf.norm(target,axis=-1)
         target = tf.math.divide(target, target_norm[:, tf.newaxis])

         prediction = tf.linalg.cross(a, b_prediction)
         prediction_norm = tf.norm(prediction,axis=-1)
         prediction = tf.math.divide(prediction, prediction_norm[:, tf.newaxis])

         loss = 0.1*tf.reduce_sum(tf.reduce_mean(tf.abs(target - prediction),axis=0))
         return loss

      losses['CP_Lead_PV'] = crossProduct_PV_Lead

      def crossProduct_PV_Sublead(**kwargs):
         targets = kwargs["targets"]
         predictions = kwargs["predictions"]
         raw = kwargs['raw_inputs']
         stdev = kwargs['stdev']
         mean = kwargs['mean']

         # De-standardising the targets & predictions
         targets = (stdev * targets) + mean
         predictions = (stdev * predictions) + mean

         px = raw[:,4]+raw[:,12]+raw[:,20]
         py = raw[:,5]+raw[:,13]+raw[:,21]
         pz = raw[:,6]+raw[:,14]+raw[:,22]
         E  = raw[:,7]+raw[:,15]+raw[:,23]

         # Formula: a x b / |a x b|
         a = tf.stack([px, py, pz], axis=-1)
         b_target = tf.stack([targets[:,4], targets[:,5], targets[:,6]], axis=-1)
         b_prediction = tf.stack([predictions[:,4], predictions[:,5], predictions[:,6]], axis=-1)

         target = tf.linalg.cross(a, b_target)
         target_norm = tf.norm(target,axis=-1)
         target = tf.math.divide(target, target_norm[:, tf.newaxis])

         prediction = tf.linalg.cross(a, b_prediction)
         prediction_norm = tf.norm(prediction,axis=-1)
         prediction = tf.math.divide(prediction, prediction_norm[:, tf.newaxis])

         loss = 0.1*tf.reduce_sum(tf.reduce_mean(tf.abs(target - prediction),axis=0))
         return loss

      losses['CP_Sublead_PV'] = crossProduct_PV_Sublead


      def crossProduct_PV(**kwargs):
         targets = kwargs["targets"]
         predictions = kwargs["predictions"]

         # Formula: a x b / |a x b|
         a_target = tf.stack([targets[:,0], targets[:,1], targets[:,2]], axis=-1)
         b_target = tf.stack([targets[:,4], targets[:,5], targets[:,6]], axis=-1)
         a_prediction = tf.stack([predictions[:,0], predictions[:,1], predictions[:,2]], axis=-1)
         b_prediction = tf.stack([predictions[:,4], predictions[:,5], predictions[:,6]], axis=-1)

         target = tf.linalg.cross(a_target, b_target)
         target_norm = tf.norm(target,axis=-1)
         target = tf.math.divide(target, target_norm[:, tf.newaxis])

         prediction = tf.linalg.cross(a_prediction, b_prediction)
         prediction_norm = tf.norm(prediction,axis=-1)
         prediction = tf.math.divide(prediction, prediction_norm[:, tf.newaxis])

         loss = 0.1*tf.reduce_sum(tf.reduce_mean(tf.abs(target - prediction),axis=0))
         return loss

      losses['CP_PV'] = crossProduct_PV


   return losses

@tf.function
def train_step(model, optimizer, x, y, y_2, z, stdev, mean, stdev_2, mean_2, loss_fns, loss_fns_2):
    with tf.GradientTape() as tape:
        # Forward pass: compute predictions using the model
        predictions_1, predictions_2 = model(x, training=True)

        # Calculate losses for each loss function in the loss_fns dictionary & Combine  individual losses into a total loss
        losses = {name: loss_fn(targets=y, predictions = predictions_1, raw_inputs = z, stdev=stdev, mean=mean) for name, loss_fn in loss_fns.items()}
        losses_2 = {name: loss_fn(targets=y_2, predictions = predictions_2, raw_inputs = z, stdev=stdev_2, mean=mean_2) for name, loss_fn in loss_fns_2.items()}

        loss_value = tf.add_n(list(losses.values()) + list(losses_2.values()))
               
    # Compute gradients with respect to trainable weights & clip them to avoid exploding gradients
    grads = tape.gradient(loss_value,model.trainable_weights)
    grads, _ = tf.clip_by_global_norm(grads,5.0)

    # Apply gradients to update model weights using the optimizer
    optimizer.apply_gradients(zip(grads,model.trainable_weights))

    return losses,losses_2,loss_value
    
@tf.function
def test_step(model, x, y, y_2, z, stdev, mean, stdev_2, mean_2, loss_fns, loss_fns_2):
    # Forward pass: compute predictions using the model
    predictions_1, predictions_2 = model(x, training=False)

    # Calculate losses for each loss function in the loss_fns dictionary & Combine  individual losses into a total loss
    losses = {name: loss_fn(targets=y,predictions = predictions_1, raw_inputs = z, stdev=stdev,mean=mean) for name, loss_fn in loss_fns.items()}
    losses_2 = {name: loss_fn(targets=y_2, predictions = predictions_2, raw_inputs = z, stdev=stdev_2, mean=mean_2) for name, loss_fn in loss_fns_2.items()}

    loss_value = tf.add_n(list(losses.values()) + list(losses_2.values()))
 
    return losses, losses_2, loss_value

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' and its parent directories created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load variables from a root file and save them to a .pkl file.")
    parser.add_argument('-t1', "--targets_1", type=str, help='A list of elements in string format, e.g., "[x, y, z]"')
    parser.add_argument('-t2', "--targets_2", type=str, help='A list of elements in string format, e.g., "[x, y, z]"')    
    parser.add_argument("-s", "--save_dir", type=str, help="The directory to save the model")
    args = parser.parse_args()

    targets_1 = args.targets_1.split(',')
    targets_2 = args.targets_2.split(',')

    inputs=[
    "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1",
    "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2",
    "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1",
    "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2",
    "ip_x_1", "ip_y_1", "ip_z_1","ipcov00_1","ipcov01_1","ipcov02_1","ipcov10_1","ipcov11_1","ipcov12_1","ipcov20_1","ipcov21_1","ipcov22_1",
    "ip_x_2", "ip_y_2", "ip_z_2","ipcov00_2","ipcov01_2","ipcov02_2","ipcov10_2","ipcov11_2","ipcov12_2","ipcov20_2","ipcov21_2","ipcov22_2",
    'ip_sig_1','ip_sig_2',
    "pt_1", "pt_2", "pt_vis",
    "metx","mety",
    ]

    dataset = "ggH"

    if args.targets_1 and args.targets_2 and args.save_dir:
       if not os.path.exists(args.save_dir):
          create_directory(args.save_dir)
       main(dataset=dataset,inputs=inputs,targets_1=targets_1,targets_2=targets_2,save_directory = args.save_dir)
    else:
       sys.exit()   
