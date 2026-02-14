import tensorflow as tf
import os
import numpy as np
from convolution_head import ConvolutionHead
from augmentation_utils import reduce_mean_mse_with_exp_weighting
import wormflow3 as wf

class End2EndWormPilot:

    def __init__(self,wm_size,conv_grad_scaling,learning_rate,curve_factor,ode_solver_unfolds=None):

        self.learning_rate = learning_rate

        self.image_width=200
        self.image_height=78

        self.x = tf.placeholder(tf.float32, shape=[None, None,self.image_height, self.image_width, 3],name='camera')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,1])

        self.conv = ConvolutionHead(num_filters=8,features_per_filter=4)
        self.feature_layer = self.conv(self.x)
        tf.identity(self.feature_layer,name='features')

        if(wm_size == "fully"):
            print("Using fully connected NPC network")
            architecture = wf.FullyConnectedWormnetArchitecture(1, num_units=19)
        elif(wm_size == "rand"):
            print("Using random NPC network")
            architecture = wf.RandomWormnetArchicture(1, num_units=19, sensory_density = 6, inter_density = 4,motor_density = 6, seed = 20190120, input_size=None)
        else:
            print("Using designed NPC network")
            architecture = wf.CommandLayerWormnetArchitectureMK2(1, num_interneurons=12, num_command_neurons=6,  sensory_density = 6, inter_density = 4, recurrency=6 ,motor_density = 6, seed = 20190120, input_size=None)


        self.wm = wf.WormnetCell(architecture)
        if(not ode_solver_unfolds is None):
            self.wm._ode_solver_unfolds = ode_solver_unfolds
        self.wm._output_mapping = wf.MappingType.Linear

        self.rnn_init_state = tf.placeholder(tf.float32,shape=[None,self.wm.state_size],name="initial_state")

        self.y,self.final_state = tf.nn.dynamic_rnn(self.wm,self.feature_layer,initial_state = self.rnn_init_state,time_major=True)

        self._sensory_neurons = self.wm._map_inputs(self.feature_layer,resuse_scope=True)

        #Output
        tf.identity(self.y,name='prediction')
        tf.identity(self.final_state,name='final_state')

        self.orig_loss = tf.reduce_mean(tf.square(tf.subtract(self.target_y, self.y)))
        self.loss = reduce_mean_mse_with_exp_weighting(y_hat=self.y,y_target=self.target_y,exp_factor=curve_factor)

        # Loss, error and training algorithm

        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y, self.target_y)))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))

        scaled_gradients = []

        for i in range(len(gradients)):
            if(gradients[i] is None):
                scaled_gradients.append(None)
            else:
                g = gradients[i]
                if("perception" in variables[i].name):
                    g = g*conv_grad_scaling
                scaled_gradients.append(g)

        scaled_gradients, _ = tf.clip_by_global_norm(scaled_gradients, 10)
        self.train_step = optimizer.apply_gradients(zip(scaled_gradients, variables))

        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.clip_op = self.wm.get_param_constrain_op()


    def get_saliency_map(self,x):
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)

        return self.conv.visual_backprop(self.sess,x)

    def share_sess(self, sess):
        self.sess = sess

    # We don't need that function at the moment, but maybe we need to
    #  visualize the features later, so let's keep it for now
    def inference_features(self, x):
        feed_dict = {
            self.x: x,
        }
        feats = self.sess.run(self.feature_layer, feed_dict)
        return feats

    def evaluate_single_sequence(self,x,y,init_state=None):
        if(init_state is None):
            init_state = np.zeros([1,self.wm.state_size])

        seq_len = x.shape[0]
        # Reshape sequence into a batch of 1 sequence
        y = y.reshape([seq_len,1,1])
        x = x.reshape([seq_len,1,x.shape[1],x.shape[2],x.shape[3]])
        feed_dict = {
            self.x: x,
            self.target_y: y,
            self.rnn_init_state: init_state}

        loss,mae,next_state= self.sess.run([self.orig_loss,self.mean_abs_error,self.final_state], feed_dict=feed_dict)
        return loss,mae,next_state

    def replay_internal_state(self,x,init_state):
        init_state = init_state.reshape([1,self.wm.state_size])

        x = x.reshape([1,1,x.shape[0],x.shape[1],x.shape[2]])
        feed_dict = {
            self.x: x,
            self.rnn_init_state: init_state}

        sensory_neurons,output,rnn_state = self.sess.run([self._sensory_neurons, self.y, self.final_state], feed_dict=feed_dict)
        return sensory_neurons.flatten(), output.flatten(),rnn_state.flatten()
        
    def train_iter(self, batch_x,batch_y):
        feed_dict = {
            self.x: batch_x,
            self.target_y: batch_y,
            self.rnn_init_state:np.zeros([batch_x.shape[1],self.wm.state_size])
            }

        (_,loss,mae) = self.sess.run([self.train_step, self.orig_loss,self.mean_abs_error], feed_dict=feed_dict)

        # Enforce parameter constraints
        self.sess.run(self.clip_op)

        return loss,mae

    def create_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, '-'+name)
        self.saver = tf.train.Saver()
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))
