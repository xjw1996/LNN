import tensorflow as tf
import os
import numpy as np
from convolution_head import ConvolutionHead
import wormflow3 as wf
from augmentation_utils import reduce_mean_mse_with_exp_weighting
from models.rnn_models import GatedRecurrentUnit, ContinuousTimeGatedRecurrentUnit,VanillaRNN,CTRNN

class UniversalRNNPilot:

    def __init__(self,num_units,conv_grad_scaling,learning_rate,curve_factor,clip_value,rnn_type,ctrnn_global_feedback,sparsity_level=0):

        self.learning_rate = learning_rate

        self.image_width=200
        self.image_height=78

        self.num_units = num_units

        self.x = tf.placeholder(tf.float32, shape=[None, None, self.image_height, self.image_width, 3],name='camera')
        self.target_y = tf.placeholder(tf.float32, shape=[None,None,1])

        self.conv = ConvolutionHead(num_filters=8,features_per_filter=4)
        self.feature_layer = self.conv(self.x)
        tf.identity(self.feature_layer,name='features')

        non_rnn_tf_vars = tf.trainable_variables()

        if(rnn_type == "ctgru"):
            self.fused_cell = ContinuousTimeGatedRecurrentUnit(self.num_units,cell_clip=clip_value)
        elif(rnn_type == "gru"):
            self.fused_cell = GatedRecurrentUnit(self.num_units,cell_clip=clip_value)
        elif(rnn_type == "tf_gru"):
            self.fused_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
        elif(rnn_type == "vanilla"):
            self.fused_cell = VanillaRNN(self.num_units)
        elif(rnn_type == "ctrnn"):
            self.fused_cell = CTRNN(self.num_units,cell_clip=clip_value,global_feedback=ctrnn_global_feedback)


        self.init_h = tf.placeholder(tf.float32,[None,self.fused_cell.state_size],name="init_h")

        rnn_out,self.final_state = tf.nn.dynamic_rnn(self.fused_cell,self.feature_layer,initial_state = self.init_h,time_major=True)

        rnn_out = tf.reshape(rnn_out,[-1,self.num_units])
        
        tf_vars = tf.trainable_variables()
        print("inside RNN all vars:")
        for v in tf_vars:
            print("Variable {}".format(str(v)))
        rnn_vars = [v for v in tf_vars if (not v in non_rnn_tf_vars)]
        print("RNN var list: ")
        for v in rnn_vars:
            print("Variable {}".format(str(v)))
        self.sparsity_level = sparsity_level
        self.sparse_op = self.sparse_op(rnn_vars,self.sparsity_level)
        
        # flatten LSTM output for dense layer to merge lstm output to the inverse_r output
        y = tf.layers.dense(rnn_out,units=1,activation=None,name='output_layer')
        # Reshape back to sequenced batch form
        self.y = tf.reshape(y,shape=[tf.shape(self.x)[0],tf.shape(self.x)[1],1])

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

    def zero_state(self,batch_size):
        return np.zeros([batch_size,self.fused_cell.state_size],dtype=np.float32)

    def get_saliency_map(self,x):
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)

        return self.conv.visual_backprop(self.sess,x)
        
    def share_sess(self, sess):
        self.sess = sess

    def sparse_op(self,rnn_vars,sparsity_level):
        if(sparsity_level <= 0 or len(rnn_vars) == 0):
            return None

        sparse_ops = []
        for v in rnn_vars:
            mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
            v_assign_op = tf.assign(v,v*mask)
            sparse_ops.append(v_assign_op)
        return sparse_ops



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
            init_state = self.zero_state(1)

        seq_len = x.shape[0]
        # Reshape sequence into a batch of 1 sequence
        y = y.reshape([seq_len,1,1])
        x = x.reshape([seq_len,1,x.shape[1],x.shape[2],x.shape[3]])
        feed_dict = {
            self.x: x,
            self.target_y: y,
            self.init_h: init_state}

        loss,mae,next_state= self.sess.run([self.orig_loss,self.mean_abs_error,self.final_state], feed_dict=feed_dict)
        return loss,mae,next_state

    def replay_internal_state(self,x,init_state):
        init_state = init_state.reshape([1,self.num_units])

        x = x.reshape([1,1,x.shape[0],x.shape[1],x.shape[2]])
        feed_dict = {
            self.x: x,
            self.init_h: init_state}

        sensory_neurons,output,rnn_state = self.sess.run([self.feature_layer, self.y, self.final_state], feed_dict=feed_dict)
        return sensory_neurons.flatten(), output.flatten(),rnn_state.flatten()
        
        
    def train_iter(self, batch_x,batch_y):
        feed_dict = {
            self.x: batch_x,
            self.target_y: batch_y,
            self.init_h:self.zero_state(batch_x.shape[1])
            }

        (_,loss,mae) = self.sess.run([self.train_step, self.orig_loss,self.mean_abs_error], feed_dict=feed_dict)

        if(not self.sparse_op is None):
            self.sess.run(self.sparse_op)

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
