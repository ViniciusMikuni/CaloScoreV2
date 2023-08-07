import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import horovod.tensorflow.keras as hvd
import argparse
import h5py as h5
import utils
from CaloScore import CaloScore
from CaloScore_distill import CaloScore_distill
from WGAN import WGAN
import gc
tf.random.set_seed(1234)

if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/FCC/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='CaloScore', help='Diffusion model to train. Options are: wgan,score,vae')
    parser.add_argument('--config', default='config_dataset2.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
    parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')
    parser.add_argument('--load', action='store_true', default=False,help='Continue baseline training')

    flags = parser.parse_args()

    config = utils.LoadJson(flags.config)
    voxels = []
    layers = []
    energies = []

    assert flags.factor%2==0 or flags.factor==1, "Distillation reduction steps needs to be even"
    assert flags.load * flags.distill == False, "Only baseline model can be loaded to continue training"   
    for dataset in config['FILES']:
        voxel_,layer_,energy_, = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            config['SHAPE'],flags.nevts,
            emax = config['EMAX'],emin = config['EMIN'],
            max_deposit=config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=config['logE'],
            rank=hvd.rank(),size=hvd.size(),
            use_1D = config['DATASET']==1,
        )
        
        voxels.append(voxel_)
        layers.append(layer_)        
        energies.append(energy_)
        

    voxels = np.reshape(voxels,config['SHAPE'])
    voxels = utils.ApplyPreprocessing(voxels,"preprocessing_{}_voxel.json".format(config['DATASET']))
    # print(np.min(voxels),np.max(voxels))
    # input()
    # voxels = utils.CalcPreprocessing(voxels,"preprocessing_{}_voxel.json".format(config['DATASET']))
    layers = np.concatenate(layers)
    layers = utils.ApplyPreprocessing(layers,"preprocessing_{}_layer.json".format(config['DATASET']))
    # layers = utils.CalcPreprocessing(layers,"preprocessing_{}_layer.json".format(config['DATASET']))

    energies = np.concatenate(energies)
    
    data_size = voxels.shape[0]
    num_cond= energies.shape[1]
    num_layer = layers.shape[1]
    
    tf_voxel = tf.data.Dataset.from_tensor_slices(voxels)
    tf_layer = tf.data.Dataset.from_tensor_slices(layers)        
    tf_energies = tf.data.Dataset.from_tensor_slices(energies)
    
    dataset = tf.data.Dataset.zip((tf_voxel, tf_layer,tf_energies))
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, voxels, tf_voxel,tf_energies,tf_layer,layers, energies
    gc.collect()
    
    BATCH_SIZE = config['BATCH']
    LR = float(config['LR'])
    NUM_EPOCHS = config['MAXEPOCH']
    EARLY_STOP = config['EARLYSTOP']
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                   patience=20, min_lr=1e-7)
    ]
    checkpoint_folder = '../checkpoints_{}_{}'.format(config['CHECKPOINT_NAME'],flags.model)
    if flags.model == 'wgan':
        num_noise=config['NOISE_DIM']
        model = WGAN(config['SHAPE'][1:],num_cond,config=config,num_noise=num_noise)
        opt_gen = tf.optimizers.RMSprop(learning_rate=LR)
        opt_dis = tf.optimizers.RMSprop(learning_rate=LR)

        opt_gen = hvd.DistributedOptimizer(
            opt_gen, backward_passes_per_step=1,
            average_aggregated_gradients=True)

        opt_dis = hvd.DistributedOptimizer(
            opt_dis, backward_passes_per_step=1,
            average_aggregated_gradients=True)
        
        model.compile(            
            d_optimizer=opt_dis,
            g_optimizer=opt_gen,
        )
        
    else:
        model = CaloScore(num_layer = num_layer,config=config)
        if flags.distill:
            if flags.factor>2:
                checkpoint_folder = '../checkpoints_{}_{}_d{}'.format(config['CHECKPOINT_NAME'],
                                                                      flags.model,flags.factor//2)
                model = CaloScore_distill(
                    model.ema_layer,model.ema_voxel,
                    factor=flags.factor//2,
                    num_layer = num_layer,
                    config = config
                )
                model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
                #previous student, now teacher
                
            else:                
                model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()

            model = CaloScore_distill(model.ema_layer,model.ema_voxel,
                                      factor=flags.factor,
                                      num_layer = num_layer,
                                      config=config,)
            if hvd.rank()==0:print("Loading Teacher from: {}".format(checkpoint_folder))
            checkpoint_folder = '../checkpoints_{}_{}_d{}'.format(config['CHECKPOINT_NAME'],flags.model,flags.factor)

        elif flags.load:
            model.load_weights('{}/checkpoint'.format(checkpoint_folder)).expect_partial()


        lr_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=config['LR']*hvd.size(),
            decay_steps=config['MAXEPOCH']*int(data_size*flags.frac/config['BATCH'])
        )

        #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        opt = hvd.DistributedOptimizer(
            opt, average_aggregated_gradients=True)


    model.compile(optimizer=opt,
                  experimental_run_tf_function=False,
                  weighted_metrics=[],
                  #run_eagerly=True
    )



    if hvd.rank()==0:
        checkpoint = ModelCheckpoint('{}/checkpoint'.format(checkpoint_folder),
                                     save_best_only=True,mode='auto',
                                     period=1,save_weights_only=True)
        callbacks.append(checkpoint)

        

    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )
