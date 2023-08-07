import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import utils
import time
# tf and friends
tf.random.set_seed(1234)

class CaloScore_distill(keras.Model):
    """Score based generative model distilled"""
    def __init__(self, teacher_layer,teacher_voxel,factor,num_layer,config=None):
        super(CaloScore_distill, self).__init__()
        
        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.factor = factor
        self.data_shape = self.config['SHAPE'][1:]
        self.num_layer = num_layer
        self.ema=0.999
        self.num_steps = 512//self.factor
        self.verbose=False

        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
        else:
            self.shape = (-1,1,1,1,1)

        
        self.loss_tracker = keras.metrics.Mean(name="loss")


        self.teacher_layer = teacher_layer
        self.teacher_voxel = teacher_voxel
        
        self.model_layer = keras.models.clone_model(teacher_layer)
        self.model_voxel = keras.models.clone_model(teacher_voxel)
        self.ema_layer = keras.models.clone_model(self.model_layer)
        self.ema_voxel = keras.models.clone_model(self.model_voxel)

        if self.verbose:
            print(self.model_voxel.summary())
        self.teacher_layer.trainable = False    
        self.teacher_voxel.trainable = False    
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_layer_tracker = keras.metrics.Mean(name="layer_loss")
        self.loss_voxel_tracker = keras.metrics.Mean(name="voxel_loss")

        if self.verbose:
            print(self.model_voxel.summary())
            
        self.factor = factor


        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker,self.loss_layer_tracker,self.loss_voxel_tracker]
    
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        return logsnr, alpha, sigma
    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    @tf.function
    def train_step(self, inputs):
        voxel,layer,cond = inputs

        i = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)

        u = (i+1) / self.num_steps
        u_mid = u - 0.5/self.num_steps
        u_s = u - 1./self.num_steps

        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(u_s)
        
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s)))
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)

        alpha_mid_reshape = tf.reshape(alpha_mid,self.shape)
        sigma_mid_reshape = tf.reshape(sigma_mid,self.shape)
        
        alpha_s_reshape = tf.reshape(alpha_s,self.shape)
        sigma_s_reshape = tf.reshape(sigma_s,self.shape)

        sigma_frac_reshape = tf.reshape(sigma_frac,self.shape)
        
        
        #voxel                        
        eps = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
        z = alpha_reshape*voxel + eps * sigma_reshape
        score_teacher = self.teacher_voxel([z,u,layer,cond],training=False)

        mean_voxel = alpha_reshape * z - sigma_reshape * score_teacher
        eps_voxel = (z - alpha_reshape * mean_voxel) / sigma_reshape
        
        z_mid = alpha_mid_reshape * mean_voxel + sigma_mid_reshape * eps_voxel
        score_teacher_mid = self.teacher_voxel([z_mid, u_mid,layer,cond],training=False)
        
        mean_voxel = alpha_mid_reshape * z_mid - sigma_mid_reshape * score_teacher_mid
        eps_voxel = (z_mid - alpha_mid_reshape * mean_voxel) / sigma_mid_reshape
        
        z_teacher = alpha_s_reshape * mean_voxel + sigma_s_reshape * eps_voxel

                
        x_target = (z_teacher - sigma_frac_reshape * z) / (alpha_s_reshape - sigma_frac_reshape * alpha_reshape)
        
        x_target = tf.where(i[:,:,None,None,None] == 0, mean_voxel, x_target)
        eps_target = (z - alpha_reshape * x_target) / sigma_reshape
        with tf.GradientTape() as tape:
            v_target = alpha_reshape * eps_target - sigma_reshape * x_target
            v = self.model_voxel([z, u,layer,cond])
            
            losses = tf.square(v - v_target)
            loss_voxel = tf.reduce_mean(losses)            
            
        g = tape.gradient(loss_voxel, self.model_voxel.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_voxel.trainable_variables))
        for weight, ema_weight in zip(self.model_voxel.weights, self.ema_voxel.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)



        #layer
        eps = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
        z = alpha*layer + eps * sigma
        score_teacher = self.teacher_layer([z, u,cond],training=False)

        mean_layer = alpha * z - sigma * score_teacher
        eps_layer = (z - alpha * mean_layer) / sigma
        
        z_mid = alpha_mid * mean_layer + sigma_mid * eps_layer
        score_teacher_mid = self.teacher_layer([z_mid, u_mid,cond],training=False)
        
        mean_layer = alpha_mid * z_mid - sigma_mid * score_teacher_mid
        eps_layer = (z_mid - alpha_mid * mean_layer) / sigma_mid
        
        z_teacher = alpha_s * mean_layer + sigma_s * eps_layer

                
        x_target = (z_teacher - sigma_frac * z) / (alpha_s - sigma_frac * alpha)
        x_target = tf.where(i == 0, mean_layer, x_target)
        eps_target = (z - alpha * x_target) / sigma
        
        with tf.GradientTape() as tape:
            v_target = alpha * eps_target - sigma * x_target
            v = self.model_layer([z, u,cond])
            
            losses = tf.square(v - v_target)
            loss_layer = tf.reduce_mean(losses)
            
            
        g = tape.gradient(loss_layer, self.model_layer.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_layer.trainable_variables))
        for weight, ema_weight in zip(self.model_layer.weights, self.ema_layer.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        
        self.loss_tracker.update_state(loss_voxel+loss_layer)
        self.loss_layer_tracker.update_state(loss_layer)
        self.loss_voxel_tracker.update_state(loss_voxel)
        
        return {
            # "loss": self.loss_tracker.result(),
            # "loss_voxel":self.loss_voxel_tracker.result(),
            # "loss_layer":self.loss_layer_tracker.result(),
            "loss": loss_voxel+loss_layer,
            "loss_voxel":loss_voxel,
            "loss_layer":loss_layer,

        }

    @tf.function
    def test_step(self, inputs):
        voxel,layer,cond = inputs

        #Define the sigma and alphas for the different time steps used in the interpolation
        i = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)

        u = (i+1) / self.num_steps
        u_mid = u - 0.5/self.num_steps
        u_s = u - 1./self.num_steps

        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(u_s)
        
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s)))
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)

        alpha_mid_reshape = tf.reshape(alpha_mid,self.shape)
        sigma_mid_reshape = tf.reshape(sigma_mid,self.shape)
        
        alpha_s_reshape = tf.reshape(alpha_s,self.shape)
        sigma_s_reshape = tf.reshape(sigma_s,self.shape)

        sigma_frac_reshape = tf.reshape(sigma_frac,self.shape)
        
        
        #voxel                        
        eps = tf.random.normal((tf.shape(voxel)),dtype=tf.float32)
        z = alpha_reshape*voxel + eps * sigma_reshape
        score_teacher = self.teacher_voxel([z,u,layer,cond],training=False)

        mean_voxel = alpha_reshape * z - sigma_reshape * score_teacher
        eps_voxel = (z - alpha_reshape * mean_voxel) / sigma_reshape
        
        z_mid = alpha_mid_reshape * mean_voxel + sigma_mid_reshape * eps_voxel
        score_teacher_mid = self.teacher_voxel([z_mid, u_mid,layer,cond],training=False)
        
        mean_voxel = alpha_mid_reshape * z_mid - sigma_mid_reshape * score_teacher_mid
        eps_voxel = (z_mid - alpha_mid_reshape * mean_voxel) / sigma_mid_reshape
        
        z_teacher = alpha_s_reshape * mean_voxel + sigma_s_reshape * eps_voxel

                
        x_target = (z_teacher - sigma_frac_reshape * z) / (alpha_s_reshape - sigma_frac_reshape * alpha_reshape)
        x_target = tf.where(i[:,:,None,None,None] == 0, mean_voxel, x_target)
        eps_target = (z - alpha_reshape * x_target) / sigma_reshape
        v_target = alpha_reshape * eps_target - sigma_reshape * x_target
        v = self.model_voxel([z, u,layer,cond])
        
        losses = tf.square(v - v_target)
        loss_voxel = tf.reduce_mean(losses)


        
        #layer
        eps = tf.random.normal((tf.shape(layer)),dtype=tf.float32)
        z = alpha*layer + eps * sigma
        score_teacher = self.teacher_layer([z, u,cond],training=False)

        mean_layer = alpha * z - sigma * score_teacher
        eps_layer = (z - alpha * mean_layer) / sigma
        
        z_mid = alpha_mid * mean_layer + sigma_mid * eps_layer
        score_teacher_mid = self.teacher_layer([z_mid, u_mid,cond],training=False)
        
        mean_layer = alpha_mid * z_mid - sigma_mid * score_teacher_mid
        eps_layer = (z_mid - alpha_mid * mean_layer) / sigma_mid
        
        z_teacher = alpha_s * mean_layer + sigma_s * eps_layer

                
        x_target = (z_teacher - sigma_frac * z) / (alpha_s - sigma_frac * alpha)
        x_target = tf.where(i == 0, mean_layer, x_target)
        eps_target = (z - alpha * x_target) / sigma
        

        v_target = alpha * eps_target - sigma * x_target
        v = self.model_layer([z, u,cond])
        
        losses = tf.square(v - v_target)
        loss_layer = tf.reduce_mean(losses)
        
        self.loss_tracker.update_state(loss_voxel+loss_layer)
        self.loss_layer_tracker.update_state(loss_layer)
        self.loss_voxel_tracker.update_state(loss_voxel)


        return {
            # "loss": self.loss_tracker.result(),
            # "loss_voxel":self.loss_voxel_tracker.result(),
            # "loss_layer":self.loss_layer_tracker.result(),

            "loss": loss_voxel+loss_layer,
            "loss_voxel":loss_voxel,
            "loss_layer":loss_layer,
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions)

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
                
            mean = alpha * x - sigma * score
            eps = (x - alpha * mean) / sigma
            x = alpha_ * mean + sigma_ * eps

        # The last step does not include any noise
        return mean

