import numpy as np
import h5py
import tensorflow as tf
from perspective_transformation import crop_all, flip_all
from augmentation_utils import draw_shadow
import os
from datetime import datetime
import cv2

class ActiveDataProvider:

    def __init__(self,h5_files,shadow_max_gamma=0,shadow_darkening_ratio=0.66,debug_flag=False):
        self.debug_flag = debug_flag
        self.shadow_max_gamma = shadow_max_gamma
        self.shadow_darkening_ratio = shadow_darkening_ratio

        self._load_h5_files(h5_files)
        self._curvature_scaling = 1000.0

    ''' Loads the h5 files that contain the recorded driving into the memory
        Input:
            @data_path: Path where the h5 files for the active test are located or a single h5 file '''
    def _load_h5_files(self,h5_files):
        self.data_x = None

        h5_files = sorted(h5_files)
        if(len(h5_files) == 0):
            raise ValueError('No .h5 files found!')

        for f in h5_files:
            # Open h5 file
            h5f = h5py.File(f,'r')

            # Convert int64 array to uint8 array, to reduce memory footprint
            cam_raw = np.array(h5f['camera_front'])
            print("Raw cam shape: ",str(cam_raw.shape))
            # Crop images immediately after loading, because we won't do augmentation (and therefore don't need the full images)
            cam_cropped = crop_all(cam_raw)

            # Cache in memory
            timestamp = np.array(h5f['timestamp'],dtype=np.float64)
            y_raw = np.array(h5f['inverse_r'])

            if(self.data_x is None):
                # Create new buffer
                self.data_x = cam_cropped
                self.data_y = y_raw
                self.data_ts = timestamp
            else:
                # Append to existing buffer
                self.data_x = np.append(self.data_x,cam_cropped,axis=0)
                self.data_y = np.append(self.data_y,y_raw,axis=0)
                self.data_ts = np.append(self.data_ts,timestamp,axis=0)

            file_date = datetime.fromtimestamp(int(timestamp[0]))

            print("Loaded file '{}' from {} containing {} images ({:0.2f} GB)".format(
                f,
                file_date.strftime("%Y-%b-%d"),
                cam_cropped.shape[0],
                cam_cropped.shape[0]*cam_cropped.shape[1]*cam_cropped.shape[2]*cam_cropped.shape[3]*1/(1024.0*1024.0*1024.0),
            ))

            # If debug flag is set, load just one file to reduce startup time
            if(self.debug_flag):
                break
            
    def _sample_random_shadow_gamma(self):
        gamma = np.random.uniform(low=0,high=self.shadow_max_gamma)
        gamma = 1+gamma
        # if coin flip is > than the ratio of darkened shadows (66%) -> make it lighter
        if(np.random.rand()>self.shadow_darkening_ratio):
            gamma = 1.0/gamma
        return gamma

    def _augment_with_shadow(self,img):
        thickness = np.random.randint(10,100)
        kernel_sizes = [3,5,7]
        blur = kernel_sizes[np.random.randint(0,len(kernel_sizes))]
        angle = np.random.uniform(low=0,high=np.pi)
        offset_x = np.random.randint(-100,100)
        offset_y = np.random.randint(-30,30)
        gamma = self._sample_random_shadow_gamma()
        img_merged = draw_shadow(img,thickness,blur,angle,offset_x,offset_y,gamma)
        return img_merged
        
    # Counts how many iterations with the given batch_size and sequence length
    # are needed to iterate over all data
    def count_epoch_size(self, batch_size, seq_len):
        return self.data_y.shape[0]//(batch_size*seq_len)

    ''' Prints information about the loaded data '''
    def summary(self,set_name="dataset"):
        print("----------------------------------------------")
        print("Summary of {}".format(set_name))

        frameskips = self.data_ts[1:] - self.data_ts[:-1]
        sampling_T = np.median(frameskips)

        total_images = self.data_x.shape[0]
        total_seconds = int(sampling_T*total_images)
        print('Total number of samples: {} ({:02d}:{:02d} at {:0.0f} Hz)'.format(
            total_images,
            total_seconds // 60,
            total_seconds % 60,
            1.0/sampling_T,
        ))

        total_memory= total_images*self.data_x.shape[1]*self.data_x.shape[2]*self.data_x.shape[3]*1/(1024.0*1024.0*1024.0)
        print('Total memory footprint of images: {0:.2f} GB'.format(total_memory))

        print("Curvature distribution (mean: {:0.2f})".format(np.mean(self.data_y)*self._curvature_scaling))
        hist,bin_edges=np.histogram(self.data_y*self._curvature_scaling, bins=[-60,-15,-5,0,5,15,60])
        hist = hist/np.sum(hist)
        for i in range(len(hist)):
            print("[{:0.2f}, {:0.2f}]: {:0.2f}%".format(
                bin_edges[i],bin_edges[i+1],
                100*hist[i]
            ))

        print("----------------------------------------------")

    ''' Shuffles the training data and iterates over it in mini-batches
        Use this function to train Feed-forward networks  '''
    def iterate_shuffled_train(self,batch_size):
        # Shuffle complete data
        p = np.random.permutation(np.arange(self.data_x.shape[0]))

        iterations_per_epoch = self.data_x.shape[0]//batch_size

        for j in range(iterations_per_epoch):
            sample_inds = np.arange(j*batch_size, batch_size*(j+1))
            inds = np.sort(p[sample_inds])
            samples = self.data_x[inds].astype(np.float32)/255.0
            labels = self.data_y[inds]
            if(self.shadow_max_gamma > 0.0):
                for s in range(samples.shape[0]):
                    samples[s] = self._augment_with_shadow(samples[s])

            # cv2.imshow("f",samples[0]/np.max(samples[0]))
            # cv2.waitKey(100)
            yield(samples,labels*self._curvature_scaling)

    ''' Creats a batch of training sequences, use this function to train RNNs
        Input: 
            @batch_size: Size of the batch 
            @sequence_length: Sequence length of each item 
        Output:
            @batched_x: numpy array of size [sequence_length,batch_size,...]
                        (Time major)
            @batched_y: numpy array of size [sequence_length,batch_size,1]
                        (Time major)
             '''
    def create_sequnced_batch(self,batch_size=32,sequence_length=16):
        # Create empty buffer that holds the input and output variable of the batch
        batched_x = np.zeros([sequence_length,batch_size,self.data_x.shape[1],self.data_x.shape[2],self.data_x.shape[3]])
        batched_y = np.zeros([sequence_length,batch_size,1])

        # Fill each item of the batch buffer
        for i in range(batch_size):
            # Sample uniformly in training data
            uniform_sample = np.random.randint(0,self.data_y.shape[0]-sequence_length)

            # Right now we don't know if the sequence is valid (does not contain a frameskip)
            is_sample_valid = False
            while(not is_sample_valid):
                # Cut out sequence of length "sequence_length"
                sequence_indices = np.arange(uniform_sample,uniform_sample+sequence_length)

                # Timestamps in seconds
                timestamps = self.data_ts[sequence_indices]
                ts_diffs = timestamps[1:]-timestamps[:-1]

                if(np.max(ts_diffs) >= 5.0):
                    # Reject sample if frameskip is more than 5 seconds
                    uniform_sample = np.random.randint(0,self.data_y.shape[0]-sequence_length)
                else:
                    is_sample_valid = True
                
            # Copy to batch buffer
            batched_x[:,i] = self.data_x[sequence_indices].astype(np.float32)/255.0
            batched_y[:,i,:] = self.data_y[sequence_indices]
            if(self.shadow_max_gamma > 0.0):
                # Augment each frame of sequences independently
                for t in range(sequence_length):
                    batched_x[t,i] = self._augment_with_shadow(batched_x[t,i])

            # cv2.imshow("f",batched_x[0,i]/np.max(batched_x[0,i]))
            # cv2.waitKey(100)

        return (batched_x,batched_y*self._curvature_scaling)

    ''' Iterates over all the data split '''
    def iterate_as_single_sequence(self,max_seq_len=16):
        # Count in how much mini-sequnces we can split the data
        number_of_chunks = self.data_x.shape[0]//max_seq_len
        # Flag indicating if there was a frameskip
        frameskip = True
        for i in range(number_of_chunks):
            sequence_index = np.arange(i*max_seq_len,(i+1)*max_seq_len)

            timestamps = self.data_ts[sequence_index]
            ts_diffs = timestamps[1:]-timestamps[:-1]

            # Remove frameskips
            if(np.max(ts_diffs) >= 5.0):
                frameskip = True
                continue

            y = self.data_y[sequence_index]
            x = self.data_x[sequence_index]

            yield (x,y*self._curvature_scaling,frameskip)
            frameskip = False
