import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import horovod.tensorflow.keras as hvd
import utils
from architectures import Unet, Resnet
import time
# tf and friends
#tf.random.set_seed(1234)

class CaloScore(keras.Model):
    """Score based generative model"""
    def __init__(self, num_layer,num_cond=1,name='SGM',config=None):
        super(CaloScore, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.num_cond = num_cond        
        self.config = config
        self.num_embed = self.config['EMBED']
        self.data_shape = self.config['SHAPE'][1:]
        self.num_layer = num_layer
        self.num_steps = 512
        self.ema=0.999
                
        
                
        self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        #Convolutional model for 3D images and dense for flatten inputs
            
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_layer_tracker = keras.metrics.Mean(name="layer_loss")
        self.loss_voxel_tracker = keras.metrics.Mean(name="voxel_loss")
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.activation = tf.keras.activations.swish

        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_layer = Input((self.num_layer))

        dense_layer = layers.Dense(self.num_embed,activation=None)(inputs_layer) 
        dense_layer = self.activation(dense_layer)

        dense_cond = layers.Dense(self.num_embed,activation=None)(inputs_cond) 
        dense_cond = self.activation(dense_cond)     

        voxel_conditional = self.Embedding(inputs_time,self.projection)
        layer_conditional = self.Embedding(inputs_time,self.projection)

        voxel_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [voxel_conditional,dense_layer,dense_cond],-1))
        
        layer_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [layer_conditional,dense_cond],-1))
        
        
        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
            use_1D = True
            nresnet_layers = 3
            resnet_dim = 128
            
            inputs,outputs = Unet(
                self.data_shape,
                voxel_conditional,
                input_embedding_dims = 16,
                stride=2,
                kernel=3,
                block_depth = 4,
                widths = [32,64,96,128],
                attentions = [False, True,True,True],
                pad=config['PAD'],
                use_1D=use_1D
            )


            
        else:
            self.shape = (-1,1,1,1,1)
            nresnet_layers = 3
            resnet_dim = 1024
            use_1D = False

            inputs,outputs = Unet(
                self.data_shape,
                voxel_conditional,
                input_embedding_dims = 32,
                stride=2,
                kernel=3,
                block_depth = 3,
                widths = [32,64,96],
                attentions = [False,False, True],
                pad=config['PAD'],
                use_1D=use_1D
            )

        
        self.model_voxel = keras.Model(inputs=[inputs,inputs_time,inputs_layer,inputs_cond],
                                       outputs=outputs)

        outputs = Resnet(
            inputs_layer,
            self.num_layer,
            layer_conditional,
            num_embed=self.num_embed,
            num_layer = nresnet_layers,
            mlp_dim= resnet_dim,
        )
        
        self.model_layer = keras.Model(inputs=[inputs_layer,inputs_time,inputs_cond],
                                       outputs=outputs)

        self.ema_layer = keras.models.clone_model(self.model_layer)
        self.ema_voxel = keras.models.clone_model(self.model_voxel)

        if self.verbose:
            print(self.model_voxel.summary())


        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker,self.loss_layer_tracker,self.loss_voxel_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb * tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def Embedding(self,inputs,projection):
        angle = inputs*projection*1000
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding
        
        
    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions)

    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    @tf.function
    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a

    
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        
        return logsnr, alpha, sigma


    

    @tf.function
    def train_step(self, inputs):
        voxel,layer,cond = inputs

        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)

        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            
        with tf.GradientTape() as tape:
            #voxel
            z = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
            perturbed_x = alpha_reshape*voxel + z * sigma_reshape
            score = self.model_voxel([perturbed_x, random_t,layer,cond])

            v = alpha_reshape * z - sigma_reshape * voxel
            losses = tf.square(score - v)
            loss_voxel = tf.reduce_mean(losses)

        trainable_variables = self.model_voxel.trainable_variables
        g = tape.gradient(loss_voxel, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_voxel.weights, self.ema_voxel.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        with tf.GradientTape() as tape:
            #layer
            z = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
            perturbed_x = alpha*layer + z * sigma            
            score = self.model_layer([perturbed_x, random_t,cond])
            v = alpha * z - sigma * layer
            losses = tf.square(score - v)
                        
            loss_layer = tf.reduce_mean(losses)
            
        trainable_variables = self.model_layer.trainable_variables
        g = tape.gradient(loss_layer, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_layer.weights, self.ema_layer.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        self.loss_tracker.update_state(loss_voxel + loss_layer)
        self.loss_layer_tracker.update_state(loss_layer)
        self.loss_voxel_tracker.update_state(loss_voxel)

        
        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":self.loss_voxel_tracker.result(),
            "loss_layer":self.loss_layer_tracker.result(),
        }

    @tf.function
    def test_step(self, inputs):
        voxel,layer,cond = inputs
        
        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
        
        
        #voxel
        z = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
        perturbed_x = alpha_reshape*voxel + z * sigma_reshape
        
        score = self.model_voxel([perturbed_x, random_t,layer,cond])
        denoise = alpha_reshape * z - sigma_reshape * voxel
        losses = tf.square(score - denoise)
        
        loss_voxel = tf.reduce_mean(losses)
        
        #layer
        z = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
        perturbed_x = alpha*layer + z * sigma            
        score = self.model_layer([perturbed_x, random_t,cond])
        denoise = alpha_reshape * z - sigma * layer
        losses = tf.square(score - denoise)
        
        loss_layer = tf.reduce_mean(losses)

        
        self.loss_tracker.update_state(loss_voxel+loss_layer)
        self.loss_layer_tracker.update_state(loss_layer)
        self.loss_voxel_tracker.update_state(loss_voxel)

        return {
            "loss": self.loss_tracker.result(),
            "loss_voxel":self.loss_voxel_tracker.result(),
            "loss_layer":self.loss_layer_tracker.result(),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def generate(self,cond):
        start = time.time()
        layer_energy = self.DDPMSampler(cond,self.ema_layer,
                                        data_shape=[self.num_layer],
                                        const_shape = [-1,1])

        voxels = self.DDPMSampler(cond,self.ema_voxel,
                                  data_shape = self.data_shape,
                                  const_shape = self.shape,
                                  layer_energy=layer_energy)
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return voxels.numpy(),layer_energy.numpy()
        

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    layer_energy=None):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0]
        data_shape = np.concatenate(([batch_size],data_shape))
        init_x = self.prior_sde(data_shape)
        
        x = init_x
        

        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)

            
            if layer_energy is None:
                score = model([x, random_t,cond],training=False)
            else:
                score = model([x, random_t,layer_energy,cond],training=False)
                alpha = tf.reshape(alpha, self.shape)
                sigma = tf.reshape(sigma, self.shape)
                alpha_ = tf.reshape(alpha_, self.shape)
                sigma_ = tf.reshape(sigma_, self.shape)


            # eps = score
            # mean = (x - sigma*eps)/alpha
                
            mean = alpha * x - sigma * score
            eps = (x - alpha * mean) / sigma
            
            x = alpha_ * mean + sigma_ * eps

        # The last step does not include any noise
        return mean

        
