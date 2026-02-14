import tensorflow as tf
import numpy as np
import time
import os
from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

# Median training time on GPU: ts 32, bs 64:
# 990 ms with sparse
# 270 ms with dense and static unfold

class WormnetCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, wormnet_architecture):
        self._architecture = wormnet_architecture

        self._input_size = -1
        self._num_units = wormnet_architecture._num_units
        self._output_size = wormnet_architecture._output_size
        self._is_built = False

        # Flag to use softplus and a mask matrix for enforcing
        # the parameter constraints
        self._implicit_param_constraints = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit

        self._input_mapping = MappingType.Affine
        self._output_mapping = MappingType.Affine

        self._erev_init_factor = 1

        self._implict_constraints = False

        self._w_init_max = 1.0
        self._w_init_min = 0.1
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1
        self._gleak_init_max = 1
        
        self._w_min_value = 0.001
        self._w_max_value = 100
        self._gleak_min_value = 0.001
        self._gleak_max_value = 100
        self._cm_t_min_value = 0.0001
        self._cm_t_max_value = 1000

        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None
        
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_size

    def _map_inputs(self,inputs,resuse_scope=False):
        # Default scope for sensory mapping
        varscope = "sensory_mapping"
        reuse = tf.AUTO_REUSE
        # Force reusing (needed to replay the sensory neurons)
        if(resuse_scope):
            varscope = self._sensory_varscope
            reuse = True

        with tf.variable_scope(varscope,reuse=reuse) as scope:
            self._sensory_varscope = scope
            if(self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear):
                w =  tf.get_variable(name='input_w',shape=[self._input_size],trainable=True,initializer=tf.initializers.constant(1))
                inputs = inputs * w
            if(self._input_mapping == MappingType.Affine):
                b =  tf.get_variable(name='input_b',shape=[self._input_size],trainable=True,initializer=tf.initializers.constant(0))
                inputs = inputs + b
        return inputs

    def _map_outputs(self,states):
        with tf.variable_scope("motor_mapping",reuse=tf.AUTO_REUSE):
            motor_neurons = tf.slice(states, [0,0], [-1,self._output_size])
            if(self._output_mapping == MappingType.Affine or self._output_mapping == MappingType.Linear):
                w =  tf.get_variable(name='output_w',shape=[self._output_size],trainable=True,initializer=tf.initializers.constant(1))
                motor_neurons = motor_neurons * w
            if(self._output_mapping == MappingType.Affine):
                b =  tf.get_variable(name='output_b',shape=[self._output_size],trainable=True,initializer=tf.initializers.constant(0))
                motor_neurons = motor_neurons + b
        return motor_neurons

    def count_params(self):
        num_of_synapses = int(np.sum(np.abs(self._adjacency_matrix)))
        num_of_sensory_synapses = int(np.sum(np.abs(self._sensory_adjacency_matrix)))

        total_parameters = 0
        if(self._fix_cm is None):
            total_parameters += self._num_units
        if(self._fix_gleak is None):
            total_parameters += self._num_units
        if(self._fix_vleak is None):
            total_parameters += self._num_units

        # Each synapse has Erev, W, mu and sigma as parameters (i.e 4 in total)
        total_parameters += 4*(num_of_sensory_synapses+num_of_synapses)

        return total_parameters

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self,input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("Wormnet"):
            if(not self._is_built):
                # TODO: Move this part into the build method inherited form tf.Layers
                self._is_built = True
                self._input_size = int(inputs.shape[-1])

                # Construct wormnet architecture 
                _input_size,_num_units,_output_size,_sensory_adjacency_matrix,_adjacency_matrix = self._architecture.get_graph(self._input_size)
                assert _input_size == self._input_size
                assert _num_units == self._num_units
                assert _output_size == self._output_size
                self._sensory_adjacency_matrix = _sensory_adjacency_matrix
                self._adjacency_matrix = _adjacency_matrix

                self._get_variables()

            elif(self._input_size != int(inputs.shape[-1])):
                raise ValueError("You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size,
                    int(inputs[-1])
                ))
            
            inputs = self._map_inputs(inputs)

            if(self._solver == ODESolver.Explicit):
                next_state = self._ode_step_explicit(inputs,state,_ode_solver_unfolds=self._ode_solver_unfolds)
            elif(self._solver == ODESolver.SemiImplicit):
                next_state = self._ode_step(inputs,state)
            elif(self._solver == ODESolver.RungeKutta):
                next_state = self._ode_step_runge_kutta(inputs,state)
            else:
                raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

            outputs = self._map_outputs(next_state)
            
        return outputs, next_state 

    # cm/t * vprev + gleak*vleak + sum w*sigmoid(vprev,mu,sigma) * Erev
    # ----------------------------------------------------------
    # cm/t + gleak + sum w * sigmoid(vprev,mu,sigma)
    def _get_variables(self):
        self.sensory_mu = tf.get_variable(name='sensory_mu',shape=[self._input_size,self._num_units],trainable=True,initializer=tf.initializers.random_uniform(minval=0.3,maxval=0.8))
        self.sensory_sigma = tf.get_variable(name='sensory_sigma',shape=[self._input_size,self._num_units],trainable=True,initializer=tf.initializers.random_uniform(minval=3.0,maxval=8.0))
        self.sensory_W = tf.get_variable(name='sensory_W',shape=[self._input_size,self._num_units],trainable=True,initializer=tf.initializers.constant(np.abs(self._sensory_adjacency_matrix)*np.random.uniform(low=self._w_init_min,high=self._w_init_max,size=[self._input_size,self._num_units])))
        self.sensory_erev = tf.get_variable(name='sensory_erev',shape=[self._input_size,self._num_units],trainable=True,initializer=tf.initializers.constant(self._sensory_adjacency_matrix*self._erev_init_factor))

        self.mu = tf.get_variable(name='mu',shape=[self._num_units,self._num_units],trainable=True,initializer=tf.initializers.random_uniform(minval=0.3,maxval=0.8))
        self.sigma = tf.get_variable(name='sigma',shape=[self._num_units,self._num_units],trainable=True,initializer=tf.initializers.random_uniform(minval=3.0,maxval=8.0))
        self.W = tf.get_variable(name='W',shape=[self._num_units,self._num_units],trainable=True,initializer=tf.initializers.constant(np.abs(self._adjacency_matrix)*np.random.uniform(low=self._w_init_min,high=self._w_init_max,size=[self._num_units,self._num_units])))
        self.erev = tf.get_variable(name='erev',shape=[self._num_units,self._num_units],trainable=True,initializer=tf.initializers.constant(self._adjacency_matrix*self._erev_init_factor))

        if(self._fix_vleak is None):
            self.vleak = tf.get_variable(name='vleak',shape=[self._num_units],trainable=True,initializer=tf.initializers.random_uniform(minval=-0.2,maxval=0.2))
        else:
            self.vleak = tf.get_variable(name='vleak',shape=[self._num_units],trainable=False,initializer=tf.initializers.constant(self._fix_vleak))

        if(self._fix_gleak is None):
            initializer=tf.initializers.constant(self._gleak_init_min)
            if(self._gleak_init_max > self._gleak_init_min):
                initializer = tf.initializers.random_uniform(minval= self._gleak_init_min,maxval = self._gleak_init_max)
                print("Init gleak in Interval: [{:0.2f}, {:0.2f}]".format(self._gleak_init_min,self._gleak_init_max))
            else:
                print("Init gleak fixed at: {:0.2f}".format(self._gleak_init_min))
            self.gleak = tf.get_variable(name='gleak',shape=[self._num_units],trainable=True,initializer=initializer)
        else:
            self.gleak = tf.get_variable(name='gleak',shape=[self._num_units],trainable=False,initializer=tf.initializers.constant(self._fix_gleak))

        if(self._fix_cm is None):
            initializer=tf.initializers.constant(self._cm_init_min)
            if(self._cm_init_max > self._cm_init_min):
                initializer = tf.initializers.random_uniform(minval= self._cm_init_min,maxval = self._cm_init_max)
                print("Init cm in Interval: [{:0.2f}, {:0.2f}]".format(self._cm_init_min,self._cm_init_max))
            else:
                print("Init cm fixed at: {:0.2f}".format(self._cm_init_min))
            self.cm_t = tf.get_variable(name='cm_t',shape=[self._num_units],trainable=True,initializer=initializer)
        else:
            self.cm_t = tf.get_variable(name='cm_t',shape=[self._num_units],trainable=False,initializer=tf.initializers.constant(self._fix_cm))

        if(self._implicit_param_constraints):
            self.W = tf.nn.softplus(self.W)
            self.sensory_W = tf.nn.softplus(self.sensory_W)
            self.gleak = tf.nn.softplus(self.gleak)
            self.cm_t = tf.nn.softplus(self.cm_t)
            
        # TODO: Maybe this should be a variable with trainable=False, such that we can restore it from a checkpoint
        self._sensory_synapse_mask = tf.constant(np.abs(self._sensory_adjacency_matrix))
        self._synapse_mask = tf.constant(np.abs(self._adjacency_matrix))

    def _ode_step(self,inputs,state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        if(self._implicit_param_constraints):
            sensory_w_activation *= self._sensory_synapse_mask 
        sensory_rev_activation = sensory_w_activation*self.sensory_erev

        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation,axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation,axis=1)

        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(self._ode_solver_unfolds):
            w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

            # If implicit parameter constraints are turned on, we need
            #  to multiply the activation by a [0,1] mask
            if(self._implicit_param_constraints):
                w_activation *= self._synapse_mask 

            rev_activation = w_activation*self.erev

            w_numerator = tf.reduce_sum(rev_activation,axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation,axis=1) + w_denominator_sensory
            
            # print('w_denominator shape: ',str(w_denominator.shape))
            # print('w_numerator shape: ',str(w_numerator.shape))

            numerator = self.cm_t * v_pre + self.gleak*self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator/denominator

        return v_pre

    def _f_prime(self,inputs,state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        if(self._implicit_param_constraints):
            sensory_w_activation *= self._sensory_synapse_mask 
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation,axis=1)


        # Unfold the mutliply ODE multiple times into one RNN step
        w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

        # If implicit parameter constraints are turned on, we need
        #  to multiply the activation by a [0,1] mask
        if(self._implicit_param_constraints):
            w_activation *= self._synapse_mask 

        w_reduced_synapse = tf.reduce_sum(w_activation,axis=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = tf.reduce_sum(sensory_in,axis=1) - v_pre*w_reduced_synapse + tf.reduce_sum(synapse_in,axis=1) - v_pre * w_reduced_sensory
        
        f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self,inputs,state):

        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h*self._f_prime(inputs,state)
            k2 = h*self._f_prime(inputs,state+k1*0.5)
            k3 = h*self._f_prime(inputs,state+k2*0.5)
            k4 = h*self._f_prime(inputs,state+k3)

            state = state + 1.0/6*(k1+2*k2+2*k3+k4)

        return state

    def _ode_step_explicit(self,inputs,state,_ode_solver_unfolds):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        if(self._implicit_param_constraints):
            sensory_w_activation *= self._sensory_synapse_mask 
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation,axis=1)


        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(_ode_solver_unfolds):
            w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

            # If implicit parameter constraints are turned on, we need
            #  to multiply the activation by a [0,1] mask
            if(self._implicit_param_constraints):
                w_activation *= self._synapse_mask 

            w_reduced_synapse = tf.reduce_sum(w_activation,axis=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = tf.reduce_sum(sensory_in,axis=1) - v_pre*w_reduced_synapse + tf.reduce_sum(synapse_in,axis=1) - v_pre * w_reduced_sensory
            
            f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre
    
    def _sigmoid(self,v_pre,mu,sigma):
        v_pre = tf.reshape(v_pre,[-1,v_pre.shape[-1],1])
        mues = v_pre - mu
        x = sigma*mues
        return tf.nn.sigmoid(x)

    def summary(self):
        print("=== Network statistics ===")
        print("# Neurons : "+str(self._num_units))

        num_of_synapses = int(np.sum(np.abs(self._adjacency_matrix)))
        num_of_sensory_synapses = int(np.sum(np.abs(self._sensory_adjacency_matrix)))

        print("# Synapses: "+str(num_of_synapses+num_of_sensory_synapses)+" ("+str(num_of_sensory_synapses)+"/"+str(num_of_synapses)+")")
        print("# Inputs  : "+str(self._input_size))
        print("# Outputs : "+str(self._output_size))

        total_parameters = 0
        if(self._fix_cm is None):
            total_parameters += self._num_units
        if(self._fix_gleak is None):
            total_parameters += self._num_units
        if(self._fix_vleak is None):
            total_parameters += self._num_units

        # Each synapse has Erev, W, mu and sigma as parameters (i.e 4 in total)
        total_parameters += 4*(num_of_sensory_synapses+num_of_synapses)

        print("# Parameters: "+str(total_parameters))

    def export_parameters(self, tf_session, path):
        # print("Exporting Parameter to ",str(path))

        if(not path.endswith(".npz") and not os.path.exists(path)):
            os.makedirs(path)

        gleak = tf_session.run(self.gleak)
        vleak = tf_session.run(self.vleak)
        cm = tf_session.run(self.cm_t)
        sigma = tf_session.run(self.sigma)
        mu = tf_session.run(self.mu)
        W = tf_session.run(self.W)
        erev = tf_session.run(self.erev)
        sensory_sigma = tf_session.run(self.sensory_sigma)
        sensory_mu = tf_session.run(self.sensory_mu)
        sensory_W = tf_session.run(self.sensory_W)
        sensory_erev = tf_session.run(self.sensory_erev)

        if(path.endswith(".npz")):
            # Save into Machine readable form
            np.savez(
                path, 
                gleak=gleak,
                vleak=vleak,
                cm=cm,
                sigma=sigma,
                mu=mu,
                W=W,
                erev=erev,
                sensory_sigma=sensory_sigma,
                sensory_mu=sensory_mu,
                sensory_W=sensory_W,
                sensory_erev=sensory_erev,
                adjacency_matrix=self._adjacency_matrix,
                sensory_adjacency_matrix=self._sensory_adjacency_matrix
            )
            return

        # Otherwise create 3 seperate textfiles

        # Save neuron parameters
        with open(os.path.join(path,'neuron_parameters.csv'),'w') as f:
            f.write("gleak;vleak;cm\n")
            for i in range(self._num_units):
                f.write("{};{};{}\n".format(gleak[i],vleak[i],cm[i])) 

        # Save synapse parameters as list
        with open(os.path.join(path,'sensory_synapses.csv'),'w') as f:
            f.write("src;dest;sigma;mu,w;erev;erev_init\n")
            for src in range(self._input_size):
                for dest in range(self._num_units):
                    if(self._sensory_adjacency_matrix[src,dest]!=0):
                        f.write("{:d};{:d};{};{};{};{};{}\n".format(
                            src,
                            dest,
                            sensory_sigma[src,dest],
                            sensory_mu[src,dest],
                            sensory_W[src,dest],
                            sensory_erev[src,dest],
                            self._sensory_adjacency_matrix[src,dest],
                        )) 
        # Save synapse parameters as list
        with open(os.path.join(path,'inter_synapses.csv'),'w') as f:
            f.write("src;dest;sigma;mu,w;erev;erev_init\n")
            for src in range(self._num_units):
                for dest in range(self._num_units):
                    if(self._adjacency_matrix[src,dest]!=0):
                        f.write("{:d};{:d};{};{};{};{};{}\n".format(
                            src,
                            dest,
                            sigma[src,dest],
                            mu[src,dest],
                            W[src,dest],
                            erev[src,dest],
                            self._adjacency_matrix[src,dest],
                        )) 


        # print("Exporting Parameter finished!")

    def get_param_constrain_op(self):
        if(self._implicit_param_constraints):
            return None
            
        cm_clipping_op = tf.assign(self.cm_t,tf.clip_by_value(self.cm_t, self._cm_t_min_value, self._cm_t_max_value))
        gleak_clipping_op = tf.assign(self.gleak,tf.clip_by_value(self.gleak, self._gleak_min_value, self._gleak_max_value))

        _sensory_synapse_mask = tf.constant(np.abs(self._sensory_adjacency_matrix))
        _synapse_mask = tf.constant(np.abs(self._adjacency_matrix))

        # Multiply W by [0,1] mask 
        # (ensures that there is no synapse from a to b, if there is none in the architecture)
        masked_w = self.W * _synapse_mask
        # Clip W by min and max range, to ensure that weight never gets negative
        w_clipping_op = tf.assign(self.W,tf.clip_by_value(masked_w, self._w_min_value, self._w_max_value))

        # Multiply W by [0,1] mask
        # (ensures that there is no synapse from a to b, if there is none in the architecture)
        masked_sensory_w = self.sensory_W * _sensory_synapse_mask
        # Clip W by min and max range, to ensure that weight never gets negative
        sensory_w_clipping_op = tf.assign(self.sensory_W ,tf.clip_by_value(masked_sensory_w, self._w_min_value, self._w_max_value))

        return [cm_clipping_op,gleak_clipping_op,w_clipping_op,sensory_w_clipping_op]

class WormnetArchicture:

    def __init__(self,output_size,num_units, input_size=None):
        self._output_size = output_size
        self._num_units = num_units
        self._adjacency_matrix = np.zeros([self._num_units,self._num_units],dtype=np.float32)

        # Specified input size here is optional
        self._input_size = input_size
        if(self._input_size != None):
            self._sensory_adjacency_matrix = np.zeros([self._input_size,self._num_units],dtype=np.float32)

    def add_synapse(self,src,dest,polarity):
        # print("Add synapse {} -> {}".format(src,dest))
        if(src < 0 or src >= self._num_units):
            raise ValueError("Source of synapse ({},{}) out of valid range (0,{})".format(src,dest,self._num_units))
        if(dest < 0 or dest >= self._num_units):
            raise ValueError("Destination of synapse ({},{}) out of valid range (0,{})".format(src,dest,self._num_units))

        self._adjacency_matrix[src,dest] = polarity

    def add_sensory_synapse(self,src,dest,polarity):
        # print("Add sensory synapse {} -> {}".format(src,dest))
        if(self._input_size is None):
            raise ValueError("Input size must be defined before sensory synapses can be added")
        if(src < 0 or src >= self._input_size):
            raise ValueError("Source of sensory synapse ({},{}) out of valid range (0,{})".format(src,dest,self._num_units))
        if(dest < 0 or dest >= self._num_units):
            raise ValueError("Destination of sensory synapse ({},{}) out of valid range (0,{})".format(src,dest,self._num_units))

        self._sensory_adjacency_matrix[src,dest] = polarity

    def construct_graph(self):
        pass

    def get_graph(self, input_size):
        if(not self._input_size is None and self._input_size != input_size):
            raise ValueError("Input size of wormnet architecture was set to {} in constructor call, and now {} was given as input size".format(self._input_size,input_size))

        # Allocate adjacency matrix of for sensory mapping
        if(self._input_size is None):
            self._input_size = input_size
            self._sensory_adjacency_matrix = np.zeros([self._input_size,self._num_units],dtype=np.float32)

        # Does nothting, can be overwritting in child classes
        self.construct_graph()

        return (self._input_size,self._num_units,self._output_size,self._sensory_adjacency_matrix,self._adjacency_matrix)


class FullyConnectedWormnetArchitecture(WormnetArchicture):

    def construct_graph(self):
        self._rng = np.random.RandomState(2018123)
        for src in range(self._input_size):
            for dest in range(self._num_units):
                polarity = 1
                if(self._rng.rand()>0.5):
                    polarity = -1
                self.add_sensory_synapse(src,dest,polarity)
        for src in range(self._num_units):
            for dest in range(self._num_units):
                if(src == dest):
                    continue
                polarity = 1
                if(self._rng.rand()>0.5):
                    polarity = -1
                self.add_synapse(src,dest,polarity)

class RandomWormnetArchicture(WormnetArchicture):

        def __init__(self,output_size,num_units, sensory_density= 3, inter_density = 8, motor_density = 4, seed = 20190120, input_size=None):
                super().__init__(output_size,num_units, input_size)
                self._sensory_density = sensory_density
                self._inter_density = inter_density
                self._motor_density = motor_density
                self._seed = seed

        def construct_graph(self):
            self._rng = np.random.RandomState(self._seed)

            # Randomly connects each sensory neuron to exactly _sensory_density number of interneurons
            for src in range(self._input_size):
                dest = self._rng.permutation(np.arange(0,self._num_units))
                for i in range(self._sensory_density):
                    polarity = 1
                    if(self._rng.rand()>0.5):
                        polarity = -1

                    self.add_sensory_synapse(src,dest[i],polarity)

            # Randomly connects each motor neuron to exactly _motor_density number of interneurons
            for dest in range(self._output_size):
                src = self._rng.permutation(np.arange(0,self._num_units))
                for i in range(self._motor_density):
                    polarity = 1
                    if(self._rng.rand()>0.5):
                        polarity = -1

                    self.add_synapse(src[i],dest,polarity)

            # Connections within interneurons
            # Create a 2D mesh of source and destination indices
            inter_neuron_range = np.arange(self._output_size,self._num_units)
            src_index,dest_index = np.meshgrid(inter_neuron_range,inter_neuron_range)
            src_index = src_index.flatten()
            dest_index = dest_index.flatten()

            # Flatten and permutate indices
            index_shuffle = self._rng.permutation(src_index.shape[0])
            src_permutation = src_index[index_shuffle]
            dest_permutation = dest_index[index_shuffle]
            
            # There is an upper bound on the number of inter-inter synapses
            for i in range(int(np.max([self._inter_density,src_index.shape[0]]))):
                polarity = 1
                if(self._rng.rand()>0.5):
                    polarity = -1
                self.add_synapse(src_permutation[i],dest_permutation[i],polarity)
            
            self._forward_reachablity_analysis()
            self._backward_reachablity_analysis()

        def _forward_reachablity_analysis(self):
            # List of interneurons that can (or cannot) be reached from at least one sensory neuron
            forward_unreachable = list(range(self._num_units))
            forward_reachable = []

            # Add all interneurons that is feed by a sensory neuron
            # to the list of reachable neurons
            for src in range(self._input_size):
                for dest in range(self._num_units):
                    if(self._sensory_adjacency_matrix[src,dest] != 0):
                        if(dest in forward_unreachable):
                            forward_unreachable.remove(dest)
                            forward_reachable.append(dest)

            # Do a forward step of reachability until the
            # the set of reachable states does not increase anymore
            reachable_count = 0
            while(reachable_count != len(forward_reachable)):
                reachable_count = len(forward_reachable)
                for src in range(self._num_units):
                    for dest in range(self._num_units):
                        if(self._adjacency_matrix[src,dest] != 0):
                            if(dest in forward_unreachable):
                                forward_unreachable.remove(dest)
                                forward_reachable.append(dest)

            # Now we have the set of reachable/non-reachable interneurons
            while len(forward_unreachable) > 0:
                dest = forward_unreachable.pop(0)

                # Connect this neuron by at least 3 other interneurons
                shuffle = self._rng.permutation(np.arange(0,len(forward_reachable)))
                for i in range(3):
                    polarity = 1
                    if(self._rng.rand()>0.5):
                        polarity = -1

                    self.add_synapse(forward_reachable[shuffle[i]],dest,polarity)


        def _backward_reachablity_analysis(self):
            # List of interneurons that can (or cannot) be reached from at least one sensory neuron
            backward_unreachable = list(range(self._num_units))
            backward_reachable = []

            # Add all motor neurons to the list of reachable neurons
            for i in range(self._output_size):
                backward_unreachable.remove(i)
                backward_reachable.append(i)

            # Do a backward step of reachability until the
            # the set of reachable states does not increase anymore
            reachable_count = 0
            while(reachable_count != len(backward_reachable)):
                reachable_count = len(backward_reachable)
                for src in range(self._num_units):
                    for dest in range(self._num_units):
                        if(self._adjacency_matrix[src,dest] != 0):
                            if(src in backward_unreachable):
                                backward_unreachable.remove(src)
                                backward_reachable.append(src)

            # Now we have the set of reachable/non-reachable interneurons
            while len(backward_unreachable) > 0:
                src = backward_unreachable.pop(0)

                # Connect this neuron by at least 3 other interneurons
                shuffle = self._rng.permutation(np.arange(0,len(backward_reachable)))
                for i in range(3):
                    polarity = 1
                    if(self._rng.rand()>0.5):
                        polarity = -1

                    self.add_synapse(src,backward_reachable[shuffle[i]],polarity)
                    



class CommandLayerWormnetArchitectureMK2(WormnetArchicture):

    # Connects each:
    # - Sensory neurons to 'sensory_density' number of interneurons
    # - Inter neuron to 'inter_density' number of command neurons
    # - Motor neuron to 'motor_density' number of command neurons
    # And also makes sure that all actually connected
    def __init__(self,
        output_size, 
        num_interneurons=6, 
        num_command_neurons=4,  
        sensory_density = 4, 
        inter_density = 2, 
        recurrency=4,
        motor_density = 4, 
        seed = 20190120, 
        input_size=None):

            super().__init__(output_size,num_interneurons+num_command_neurons + output_size, input_size)
            self._num_interneurons = num_interneurons
            self._num_command_neurons = num_command_neurons
            self._sensory_density = sensory_density
            self._inter_density = inter_density
            self._recurrency = recurrency
            self._motor_density = motor_density
            self._seed = seed
            self._verbose = False

            self._prob_excitatory = 0.7
            if(self._motor_density > self._num_command_neurons):
                raise ValueError("Motor density must be less or equal than the number of command neurons")
            if(self._sensory_density > self._num_interneurons):
                raise ValueError("Sensory density must be less or equal than the number of interneurons neurons")
            if(self._inter_density > self._num_command_neurons):
                raise ValueError("Inter density must be less or equal than the number of command neurons")

    def _connect_sensory_inter(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size + self._num_command_neurons
        real_src = src
        if(self._verbose):
            print("Senosry ({}) -> Inter ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_sensory_synapse(real_src,real_dest,polarity)

    def _connect_inter_command(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src  + self._output_size + self._num_command_neurons
        if(self._verbose):
            print("Inter ({}) -> Command ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)

    def _connect_command_command(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src  + self._output_size
        if(self._verbose):
            print("Command ({}) -> Command ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)

    def _connect_command_motor(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest
        real_src = src  + self._output_size
        if(self._verbose):
            print("Command ({}) -> Motor ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)


    def construct_graph(self):
        self._rng = np.random.RandomState(self._seed)
        ##################### Sensory layer ########################
        unreachable_interneurons = list(range(self._num_interneurons))
        # Randomly connects each sensory neuron to exactly _sensory_density number of interneurons
        for src in range(self._input_size):
            dest_index = self._rng.permutation(np.arange(0,self._num_interneurons))
            for i in range(self._sensory_density):
                
                if(dest_index[i] in unreachable_interneurons):
                    unreachable_interneurons.remove(dest_index[i])
                self._connect_sensory_inter(src,dest_index[i])

        # If it happens that some interneurons are not connected, connect them now
        mean_interneuron_fanin = int(self._input_size*self._sensory_density/self._num_interneurons)
        for i in unreachable_interneurons:
            src = self._rng.permutation(np.arange(0,self._input_size))
            for j in range(mean_interneuron_fanin):
                self._connect_sensory_inter(src[j],i)
        
        ################# Inter layer ########################
        # Randomly connect interneurons to command neurons
        unreachable_commandneurons = list(range(self._num_command_neurons))
        for src_index in range(self._num_interneurons):
            dest_index = self._rng.permutation(np.arange(0,self._num_command_neurons))
            for i in range(self._inter_density):
                
                if(dest_index[i] in unreachable_commandneurons):
                    unreachable_commandneurons.remove(dest_index[i])
                
                self._connect_inter_command(src_index,dest_index[i])

        # If it happens that some commandneurons are not connected, connect them now
        mean_commandneurons_fanin = int(self._num_interneurons*self._inter_density/self._num_command_neurons)
        for i in unreachable_commandneurons:
            src_index = self._rng.permutation(np.arange(0,self._num_interneurons))
            for j in range(mean_commandneurons_fanin):

                self._connect_inter_command(src_index[j],i)
            
        ################# Command layer ########################
        # Add recurrency in command neurons 
        for i in range(self._recurrency):
            src = self._rng.randint(0,self._num_command_neurons)
            dest = self._rng.randint(0,self._num_command_neurons)
            
            self._connect_command_command(src,dest)

        ################# Motor layer ########################
        # Randomly connect command neurons to motor neurons
        unreachable_commandneurons = list(range(self._num_command_neurons))
        for dest in range(self._output_size):
            src_index = self._rng.permutation(np.arange(0,self._num_command_neurons))
            for i in range(self._motor_density):

                if(src_index[i] in unreachable_commandneurons):
                    unreachable_commandneurons.remove(src_index[i])
                
                self._connect_command_motor(src_index[i],dest)

        # If it happens that some commandneurons are not connected, connect them now
        mean_motorneuron_fanin = int(self._output_size*self._motor_density/self._num_command_neurons)
        for i in unreachable_commandneurons:
            dest = self._rng.permutation(np.arange(0,self._output_size))
            for j in range(mean_motorneuron_fanin):

                self._connect_command_motor(i,dest[j])



class CommandLayerWormnetArchitecture(WormnetArchicture):

    # Connects each:
    # - Sensory neurons to 'sensory_density' number of interneurons
    # - Inter neuron to 'inter_density' number of command neurons
    # - Motor neuron to 'motor_density' number of command neurons
    # And also makes sure that all actually connected
    def __init__(self,
        output_size, 
        num_interneurons=6, 
        num_command_neurons=4,  
        sensory_density = 4, 
        inter_density = 2, 
        recurrency=4,
        motor_density = 4, 
        seed = 20190120, 
        input_size=None):

            super().__init__(output_size,num_interneurons+num_command_neurons + output_size, input_size)
            self._num_interneurons = num_command_neurons
            self._num_command_neurons = num_interneurons
            self._sensory_density = sensory_density
            self._inter_density = inter_density
            self._recurrency = recurrency
            self._motor_density = motor_density
            self._seed = seed
            self._verbose = False

            self._prob_excitatory = 0.7
            if(self._motor_density > self._num_command_neurons):
                raise ValueError("Motor density must be less or equal than the number of command neurons")
            if(self._sensory_density > self._num_interneurons):
                raise ValueError("Sensory density must be less or equal than the number of interneurons neurons")
            if(self._inter_density > self._num_command_neurons):
                raise ValueError("Inter density must be less or equal than the number of command neurons")

    def _connect_sensory_inter(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size + self._num_command_neurons
        real_src = src
        if(self._verbose):
            print("Senosry ({}) -> Inter ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_sensory_synapse(real_src,real_dest,polarity)

    def _connect_inter_command(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src  + self._output_size + self._num_command_neurons
        if(self._verbose):
            print("Inter ({}) -> Command ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)

    def _connect_command_command(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src  + self._output_size
        if(self._verbose):
            print("Command ({}) -> Command ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)

    def _connect_command_motor(self,src,dest,polarity = None):
        if(polarity is None):
            polarity = 1
            if(self._rng.rand()>self._prob_excitatory):
                polarity = -1
        real_dest = dest
        real_src = src  + self._output_size
        if(self._verbose):
            print("Command ({}) -> Motor ({}) [{} -> {}]".format(src,dest,real_src,real_dest))
        self.add_synapse(real_src,real_dest,polarity)


    def construct_graph(self):
        self._rng = np.random.RandomState(self._seed)

        unreachable_interneurons = list(range(self._num_interneurons))
        # Randomly connects each sensory neuron to exactly _sensory_density number of interneurons
        for src in range(self._input_size):
            dest_index = self._rng.permutation(np.arange(0,self._num_interneurons))
            for i in range(self._sensory_density):
                
                if(dest_index[i] in unreachable_interneurons):
                    unreachable_interneurons.remove(dest_index[i])
                self._connect_sensory_inter(src,dest_index[i])

        # If it happens that some interneurons are not connected, connect them now
        mean_interneuron_fanin = int(self._input_size*self._sensory_density/self._num_interneurons)
        for i in unreachable_interneurons:
            src = self._rng.permutation(np.arange(0,self._input_size))
            for j in range(mean_interneuron_fanin):
                self._connect_sensory_inter(src[j],i)
        
        # Randomly connect interneurons to command neurons
        unreachable_commandneurons = list(range(self._num_command_neurons))
        for src_index in range(self._inter_density):
            dest_index = self._rng.permutation(np.arange(0,self._num_command_neurons))
            for i in range(self._inter_density):
                
                if(dest_index[i] in unreachable_commandneurons):
                    unreachable_commandneurons.remove(dest_index[i])
                
                self._connect_inter_command(src_index,dest_index[i])

        # If it happens that some commandneurons are not connected, connect them now
        mean_commandneurons_fanin = int(self._num_interneurons*self._inter_density/self._num_command_neurons)
        for i in unreachable_commandneurons:
            src_index = self._rng.permutation(np.arange(0,self._num_interneurons))
            for j in range(mean_commandneurons_fanin):

                self._connect_inter_command(src_index[j],i)
            
        # Add recurrency in command neurons 
        for i in range(self._recurrency):
            src = self._rng.randint(0,self._num_command_neurons)
            dest = self._rng.randint(0,self._num_command_neurons)
            
            self._connect_command_command(src,dest)

        # Randomly connect command neurons to motor neurons
        unreachable_commandneurons = list(range(self._num_command_neurons))
        for dest in range(self._output_size):
            src_index = self._rng.permutation(np.arange(0,self._num_command_neurons))
            for i in range(self._motor_density):

                if(src_index[i] in unreachable_commandneurons):
                    unreachable_commandneurons.remove(src_index[i])
                
                self._connect_command_motor(src_index[i],dest)

        # If it happens that some commandneurons are not connected, connect them now
        mean_motorneuron_fanin = int(self._output_size*self._motor_density/self._num_command_neurons)
        for i in unreachable_commandneurons:
            dest = self._rng.permutation(np.arange(0,self._output_size))
            for j in range(mean_motorneuron_fanin):

                self._connect_command_motor(i,dest[j])


if(__name__ == '__main__'):
    # architecture = FullyConnectedWormnetArchitecture(1,8)
    architecture = CommandLayerWormnetArchitectureMK2(1, num_interneurons=8, num_command_neurons=4,  sensory_density = 4, inter_density = 3, recurrency=2 ,motor_density = 4, seed = 20190120, input_size=None)
    # architecture = RandomWormnetArchicture(1,10, sensory_density= 5, inter_density = 8, motor_density = 4, seed = 20190120, input_size=None)
    # architecture = WormnetArchicture(output_size=1,num_units=5,input_size=2)
    # architecture.add_sensory_synapse(0,1,1)
    # architecture.add_sensory_synapse(0,2,1)
    # architecture.add_sensory_synapse(0,3,1)
    # architecture.add_sensory_synapse(1,2,-1)
    # architecture.add_sensory_synapse(1,3,-1)
    # architecture.add_sensory_synapse(1,4,-1)

    # architecture.add_synapse(1,0,1)
    # architecture.add_synapse(2,0,-1)
    # architecture.add_synapse(3,0,1)
    # architecture.add_synapse(4,0,-1)
    # architecture.add_synapse(0,1,-1)
    # architecture.add_synapse(0,4,-1)
    wm = WormnetCell(architecture)

    time_dimension = 100
    num_of_observations = 2
    batch_size = 1
    
    rnn_init_state = tf.placeholder(tf.float32,shape=[None,wm.state_size])

    x = tf.placeholder(tf.float32,shape=[time_dimension,None,num_of_observations])
    # Unstack time dimension

    # sinusoid
    # sinusoid
    train_x = np.array([np.sin(np.linspace(0,4*np.pi,time_dimension)), np.cos(np.linspace(0,4*np.pi,time_dimension))])
    train_y = np.array(np.sin(np.linspace(0,8*np.pi,time_dimension)))

    train_x = np.transpose(train_x,axes=[1,0]).reshape([time_dimension, 1,2])
    train_y = train_y.reshape([time_dimension,1,1])

    print("Train_x shape: "+str(train_x.shape))
    print("Train_y shape: "+str(train_y.shape))


    build_start = time.time()

    # unstacked_signal = tf.unstack(x,axis=0)
    # wm_out,_ = tf.contrib.rnn.static_rnn(wm,unstacked_signal,initial_state=rnn_init_state)
    # wm_out = tf.stack(wm_out)

    wm_out,_ = tf.nn.dynamic_rnn(wm,x,initial_state = rnn_init_state,time_major=True)

    ys = tf.placeholder(shape=[None,None,1], dtype=tf.float32,name="supervised_output")
    loss = tf.reduce_mean(tf.square(tf.subtract(ys, wm_out)))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    clip_op = wm.get_param_constrain_op()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    build_time = time.time() - build_start
    print("Graph build time: {:0.2f} s".format(build_time))

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    max_num_epochs = 401
    train_times = []
    for epoch in range(max_num_epochs):
        start = time.time()
        feed_dict = {
            x:train_x,
            ys:train_y,
            rnn_init_state: np.zeros([batch_size,wm.state_size])
        }
        train_loss,_,current_output = sess.run([loss,train_step,wm_out],feed_dict=feed_dict)

        sess.run(clip_op)

        if(epoch % 20 == 0):
            if(not os.path.exists('sine_traces')):
                os.makedirs('sine_traces')
            
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            sns.set()
            plt.figure(figsize=(6,5))
            plt.plot(current_output[:,0,0],label='wormnet')
            plt.plot(train_y[:,0,0],linestyle='dashed',label='label signal')
            plt.xlabel('Time steps')
            plt.ylabel('Neuron potential')
            plt.title("Neuron 0")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('sine_traces','epochs_{:04d}.png'.format(epoch)))
            plt.close()

            print('epoch {} Train loss: {:0.2f}'.format(epoch,train_loss))


