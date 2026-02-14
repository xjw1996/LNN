import tensorflow as tf
import os
import numpy as np
import cv2
from augmentation_utils import reduce_mean_mse_with_exp_weighting

class CnnModel:

    def __init__(self,learning_rate,curve_factor,drf,dr1,dr2):

        def conv2d(x, W,stride=1):
          """conv2d returns a 2d convolution layer with full stride."""
          return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


        def weight_variable(shape):
          """weight_variable generates a weight variable of a given shape."""
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)


        def bias_variable(shape):
          """bias_variable generates a bias variable of a given shape."""
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        self.drf = drf
        self.dr1 = dr1
        self.dr2 = dr2

        self.image_width=200
        self.image_height=78

        self.kernel_dim_1 = 5
        self.kernel_dim_2 = 5
        self.kernel_dim_3 = 3
        self.kernel_dim_4 = 3
        self.kernel_dim_5 = 3

        self.stride_1 = 2
        self.stride_2 = 2
        self.stride_3 = 2
        self.stride_4 = 1
        self.stride_5 = 1

        self.num_filters1 = 24
        self.num_filters2 = 36
        self.num_filters3 = 48
        self.num_filters4 = 64
        self.num_filters5 = 64


        self.fc_size_1 = 1000
        self.fc_size_2 = 100

        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, 3],name='camera')
        self.y = tf.placeholder(tf.float32, shape=[None,1])

        # Do Image whitening (Standardization)
        self.x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x)

        # First convolutional layer
        self.W_conv1 = weight_variable([self.kernel_dim_1, self.kernel_dim_1, 3, self.num_filters1])
        self.b_conv1 = bias_variable([self.num_filters1])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1,self.stride_1) + self.b_conv1)

        # Second convolutional layer
        self.W_conv2 = weight_variable([self.kernel_dim_2, self.kernel_dim_2, self.num_filters1, self.num_filters2])
        self.b_conv2 = bias_variable([self.num_filters2])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2,self.stride_2) + self.b_conv2)

        # Third convolutional layer
        self.W_conv3 = weight_variable([self.kernel_dim_3, self.kernel_dim_3, self.num_filters2, self.num_filters3])
        self.b_conv3 = bias_variable([self.num_filters3])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3,self.stride_3) + self.b_conv3)

        # Fourth convolutional layer
        self.W_conv4 = weight_variable([self.kernel_dim_4, self.kernel_dim_4, self.num_filters3, self.num_filters4])
        self.b_conv4 = bias_variable([self.num_filters4])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4,self.stride_4) + self.b_conv4)

        # Fifth convolutional layer
        self.W_conv5 = weight_variable([self.kernel_dim_5, self.kernel_dim_5, self.num_filters4, self.num_filters5])
        self.b_conv5 = bias_variable([self.num_filters5])
        self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5,self.stride_5) + self.b_conv5)
        self.h_out5 = self.h_conv5

        print("Last convolutional map is of shape "+str(self.h_out5.shape))

        self.keep_prob_flattened = tf.placeholder(tf.float32,name='dropout_f')
        self.keep_prob_fc1 = tf.placeholder(tf.float32,name='dropout1')
        self.keep_prob_fc2 = tf.placeholder(tf.float32,name='dropout2')

        # Flatten convolution output
        flat_size = int(self.h_out5.shape[1])*int(self.h_out5.shape[2])*int(self.h_out5.shape[3])
        print('Flat size: '+str(flat_size))
        self.filter_output  = tf.reshape(self.h_out5,[-1,flat_size])
        self.filter_output = tf.nn.dropout(self.filter_output,self.keep_prob_flattened)

        # Fully connected layer 1
        self.W_fc1 = weight_variable([flat_size, self.fc_size_1])
        self.b_fc1 = bias_variable([self.fc_size_1])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.filter_output, self.W_fc1) + self.b_fc1)
        self.h_fc1= tf.nn.dropout(self.h_fc1,self.keep_prob_fc1)

        # Fully connected layer 2
        self.W_fc2 = weight_variable([self.fc_size_1, self.fc_size_2])
        self.b_fc2 = bias_variable([self.fc_size_2])
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
        self.h_fc2 = tf.nn.dropout(self.h_fc2,self.keep_prob_fc2)

        # Output Layer
        self.W_fc3 = weight_variable([self.fc_size_2, 1])
        # No bias at the output layer
        # self.b_fc3 = bias_variable([1])
        # self.y_ =  tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3
        self.y_ =  tf.matmul(self.h_fc2, self.W_fc3)
        tf.identity(self.y_,name='prediction')

        # Loss, error and training algorithm
        self.orig_loss = tf.reduce_mean(tf.square(tf.subtract(self.y_, self.y)))
        self.loss = reduce_mean_mse_with_exp_weighting(y_hat=self.y_,y_target=self.y,exp_factor=curve_factor)

        self.mean_abs_error = tf.reduce_mean(tf.abs(tf.subtract(self.y_, self.y)))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def share_sess(self, sess):
        self.sess = sess

    def get_saliency_map(self,x):
        x = np.expand_dims(x, axis=0)

        return self.visual_backprop(self.sess,x)

    def replay_internal_state(self,x,init_state):
        x = x.reshape([1,x.shape[0],x.shape[1],x.shape[2]])
        feed_dict = {
            self.x: x,
            self.keep_prob_flattened: 1.0,
            self.keep_prob_fc1: 1.0,
            self.keep_prob_fc2: 1.0
        }

        output,h1,h2 = self.sess.run([self.y_,self.h_fc1,self.h_fc2], feed_dict=feed_dict)
        hidden = np.concatenate([h1.flatten(),h2.flatten()])
        return hidden, output.flatten(),None


    def evaluate_single_sequence(self, x,y,init_states):
        # Input data is already a sequence (=batch), so we don't need
        # any reshaping
        loss,mae = self.evaluate(x,y)
        return loss,mae,None

    def evaluate(self, x,y):
        feed_dict = {self.x: x,
                     self.y: y,
                     self.keep_prob_flattened: 1.0,
                     self.keep_prob_fc1: 1.0,
                     self.keep_prob_fc2: 1.0}

        (loss,mae) = self.sess.run([self.orig_loss,self.mean_abs_error], feed_dict=feed_dict)
        return loss,mae
        
    def train_iter(self, batch_x,batch_y):
        feed_dict = {self.x: batch_x,
                     self.y: batch_y,
                     self.keep_prob_flattened: self.drf,
                     self.keep_prob_fc1: self.dr1,
                     self.keep_prob_fc2: self.dr2}

        loss,mae,_ = self.sess.run([self.orig_loss,self.mean_abs_error,self.train_step], feed_dict=feed_dict)
        return loss,mae

    def predict(self, feed):
        y = self.sess.run(self.y_, feed_dict=feed)
        return y

    def create_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, '-'+name)
        # Create a new saver object
        self.saver = tf.train.Saver()
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore_from_checkpoint(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))
        
    # Generates saliency mask according to VisualBackprop (https://arxiv.org/abs/1611.05418).

    def visual_backprop(self,tf_session,x_value):

        conv_layers = [self.h_conv1,self.h_conv2,self.h_conv3,self.h_conv4,self.h_conv5]
        A = tf_session.run(conv_layers, feed_dict={self.x: x_value})

        means = []
        aux_list = []
        for i in range(len(A)): #for each feature map
            # layer index, batch_dimension
            means.append( np.mean( A[i][0], 2 ) )

        for i in range(len(means)-2, -1, -1):
            smaller = means[i+1]
            aux_list.append(("layer_{:d}".format(i),smaller))
            scaled_up = cv2.resize(smaller, (means[i].shape[::-1]))
            means[i] = np.multiply(means[i],scaled_up)

        mask = means[0]
        mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
        # mask = np.exp(mask)
        # mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
        mask = np.clip(mask, 0,1)


        return mask,aux_list
