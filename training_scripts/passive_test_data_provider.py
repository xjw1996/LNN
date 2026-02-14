import numpy as np
import h5py
import time
import tensorflow as tf
from perspective_transformation import crop_all
from augmentation_utils import draw_shadow
import os
import datetime
import threading
import cv2

class PassiveTestDataProvider:

    experiment_split_table = {
        0: ['20180705-203123_blue_prius_mit_to_devens.h5'],
        1: ['20180714-174053_blue_prius_devens_weston.h5'],
        2: ['20180719-164626_blue_prius_devens_to_mit.h5'],
        3: ['20180721-190255_blue_prius_waltham_to_mit.h5'],
        4: ['20180722-184252_blue_prius_mit_to_weston.h5'],
        5: ['20180726-114301_blue_prius_mit_to_devens.h5'],
        6: ['20180726-123104_blue_prius_ayer_to_devens.h5' ,'20180721-181236_blue_prius_weston_to_waltham.h5'],
        7: ['20180707-103601_blue_prius_everett.h5','20180721-105355_blue_prius_mit_to_weston.h5'],
        8: ['20180707-115905_blue_prius_cambridge_to_mit.h5','20180715-184430_blue_prius_weston_cambridge.h5'],
        9: ['20180728-120843_blue_prius_mit_to_everett.h5','20180707-113533_blue_prius_everett_to_cambridge.h5'],
    }
    def __init__(self,h5_files_directory,experiment_id,shadow_max_gamma,shadow_darkening_ratio,debug_flag=False):
        if(experiment_id < 0 or experiment_id >= len(self.experiment_split_table)):
            raise ValueError('experiment_id out of bound! Must be within 0 and 9')

        self.experiment_id = experiment_id
        self.debug_flag = debug_flag
        self._curvature_scaling = 1000.0
        self.shadow_max_gamma = shadow_max_gamma
        self.shadow_darkening_ratio = shadow_darkening_ratio

        self._write_lock = threading.Lock()
        self._load_h5_files(h5_files_directory)

    def _append_or_create(self,x,y,ts, add_x,add_y,add_ts):
        if(x is None):
            # Create new buffer
            x = add_x
            y = add_y
            ts = add_ts
        else:
            # Append to existing buffer
            x = np.append(x,add_x,axis=0)
            y = np.append(y,add_y,axis=0)
            ts = np.append(ts,add_ts,axis=0)
        return (x,y,ts)
            
    def _cut_beginning_and_end(self,x,y,ts):
        slice_size = int(0.05*x.shape[0])
        # Make slice size a multiple of 16
        slice_size += 16 - slice_size % 16

        begin_x = x[:slice_size]
        begin_y = y[:slice_size]
        begin_ts = ts[:slice_size]
        end_x = x[-slice_size:]
        end_y = y[-slice_size:]
        end_ts = ts[-slice_size:]

        valid_x = np.concatenate([begin_x,end_x],axis=0)
        valid_y = np.concatenate([begin_y,end_y],axis=0)
        valid_ts = np.concatenate([begin_ts,end_ts],axis=0)

        x = x[slice_size:-slice_size]
        y = y[slice_size:-slice_size]
        ts = ts[slice_size:-slice_size]

        return (valid_x,valid_y,valid_ts,x,y,ts)

    ''' Loads the h5 files that contain the recorded driving into the memory
        Input:
            @data_path: Path where the h5 files for the passive test are located '''
    def _load_h5_files(self,data_path):
        self.train_x = None
        self.train_y = None
        self.train_ts = None
        self.valid_x = None
        self.valid_y = None
        self.valid_ts = None
        self.test_x = None
        self.test_y = None
        self.test_ts = None

        h5_files = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')])
        if(len(h5_files) == 0):
            raise ValueError('No .h5 files found!')

        if(self.debug_flag):
            # use only this file for debugging, it has onlly 600 MB
            # h5_files = ["20180707-103601_blue_prius_everett.h5"]
            # 1.3 GB
            h5_files = ["20180728-131056_blue_prius_everett_to_mit.h5"]            


        for f in h5_files:
            # Open h5 file
            h5f = h5py.File(os.path.join(data_path, f),'r')
            print("Loading: ",str(f))
            # Convert int64 array to uint8 array, to reduce memory footprint
            cam_raw = np.array(h5f['camera_front'],dtype=np.uint8)
            # Crop images immediately after loading, because we won't do augmentation (and therefore don't need the full images)
            cam_cropped = crop_all(cam_raw)

            # Cache in memory
            timestamp = np.array(h5f['timestamp'],dtype=np.float64)
            y_raw = np.array(h5f['inverse_r'])

            # If debug flag is set, load just one file to reduce startup time
            if(self.debug_flag):
                self.train_x = cam_cropped
                self.train_y = y_raw
                self.train_ts = timestamp
                self.valid_x = cam_cropped
                self.valid_y = y_raw
                self.valid_ts = timestamp
                self.test_x = cam_cropped
                self.test_y = y_raw
                self.test_ts = timestamp
                print("Debug break")
                break

            if(f in self.experiment_split_table[self.experiment_id]):
                # Validation file
                use_type = 'Validation'
                self.test_x,self.test_y,self.test_ts = self._append_or_create(self.test_x,self.test_y,self.test_ts,cam_cropped,y_raw,timestamp)
            else:
                # Training file
                use_type = 'Training'
                valid_x,valid_y,valid_ts,train_x,train_y,train_ts = self._cut_beginning_and_end(cam_cropped,y_raw,timestamp)
                self.train_x,self.train_y,self.train_ts = self._append_or_create(self.train_x,self.train_y,self.train_ts,train_x,train_y,train_ts)
                self.valid_x,self.valid_y,self.valid_ts = self._append_or_create(self.valid_x,self.valid_y,self.valid_ts,valid_x,valid_y,valid_ts)

            print('Loaded file "'+f+'" containing '+str(cam_cropped.shape[0])+' images ({:0.2f} GB) for [{}]'.format(
                cam_cropped.shape[0]*cam_cropped.shape[1]*cam_cropped.shape[2]*cam_cropped.shape[3]*1/(1024.0*1024.0*1024.0),use_type))
            
    def count_training_set(self, batch_size, seq_len):
        return self.train_y.shape[0]//(batch_size*seq_len)

    ''' Prints information about the loaded data '''
    def summary(self):
        total_images = self.train_x.shape[0]+self.valid_x.shape[0]+self.test_x.shape[0]
        print('Total number samples: {}'.format(total_images))
        print(' Training  : {} ({:.2f} %)'.format(self.train_x.shape[0],100*self.train_x.shape[0]/total_images))
        print(' Validation: {} ({:.2f} %)'.format(self.valid_x.shape[0],100*self.valid_x.shape[0]/total_images))
        print(' Test      : {} ({:.2f} %)'.format(self.test_x.shape[0],100*self.test_x.shape[0]/total_images))

        total_memory= total_images*self.train_x.shape[1]*self.train_x.shape[2]*self.train_x.shape[3]*1/(1024.0*1024.0*1024.0)
        print('Total memory footprint of images: {0:.2f} GB'.format(total_memory))

        print("Curvature distribution of training set (mean: {:0.2f})".format(np.mean(self.train_y)*self._curvature_scaling))
        hist,bin_edges=np.histogram(self.train_y*self._curvature_scaling, bins=[-60,-15,-5,0,5,15,60])
        hist = hist/np.sum(hist)
        for i in range(len(hist)):
            print("[{:0.2f}, {:0.2f}]: {:0.2f}%".format(
                bin_edges[i],bin_edges[i+1],
                100*hist[i]
            ))
        print("----------------------------------------------")

    def _sample_random_shadow_gamma(self):
        gamma = np.random.uniform(low=0,high=self.shadow_max_gamma)
        gamma = 1+gamma
        # if coin flip is > than the ratio of darkened shadows (66%) -> make it lighter
        if(np.random.rand()>self.shadow_darkening_ratio):
            gamma = 1.0/gamma
        return gamma

    def concurrent_map_shadow(self,images):
        N = images.shape[0]
        result = np.empty(shape=images.shape,dtype=np.float32)

        # wrapper to dispose the result in the right slot
        def task_wrapper_single(i):
            tmp = self._augment_single_image_with_shadow(images[i])
            self._write_lock.acquire()
            result[i] = tmp
            self._write_lock.release()

        # def task_wrapper(start_idx,n):
            # Process array
            # tmp = []
            # for i in range(n):
            #     tmp.append(self._augment_single_image_with_shadow(images[start_idx+i]))
            
            # # Write
            # self._write_lock.acquire()
            # for i in range(n):
            #     result[start_idx+i] = tmp[i]
            #     # print("write: {} ".format(start_idx+i))
            # self._write_lock.release()

        cv2.setNumThreads(1)
        # num_threads = 6
        # images_per_thread = int(np.ceil(N/num_threads))
        # # images_per_thread = N//num_threads
        # start_idx = [i*images_per_thread for i in range(num_threads)]
        # n_idx = [images_per_thread for i in range(num_threads)]
        # # remaining items
        # n_idx[-1] = N-(num_threads-1)*images_per_thread

        # threads = [threading.Thread(target=task_wrapper, args=(start_idx[i],n_idx[i],)) for i in range(num_threads)]
        threads = [threading.Thread(target=task_wrapper_single, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # for i in range(N):
        #     cv2.imshow("f",result[i]/np.max(result[i]))
        #     cv2.waitKey(100)

        return result

    def _augment_single_image_with_shadow(self,img):
        thickness = np.random.randint(10,100)
        kernel_sizes = [3,5,7]
        blur = kernel_sizes[np.random.randint(0,len(kernel_sizes))]
        angle = np.random.uniform(low=0,high=np.pi)
        offset_x = np.random.randint(-100,100)
        offset_y = np.random.randint(-30,30)
        gamma = self._sample_random_shadow_gamma()
        img_merged = draw_shadow(img,thickness,blur,angle,offset_x,offset_y,gamma)
        return img_merged
        
    ''' Shuffles the training data and iterates over it in mini-batches
        Use this function to train Feed-forward networks  '''
    def iterate_shuffled_train(self,batch_size):
        # Shuffle complete data
        p = np.random.permutation(np.arange(self.train_x.shape[0]))

        iterations_per_epoch = self.train_x.shape[0]//batch_size

        for j in range(iterations_per_epoch):
            sample_inds = np.arange(j*batch_size, batch_size*(j+1))
            inds = np.sort(p[sample_inds])
            samples = self.train_x[inds].astype(np.float32)/255.0
            labels = self.train_y[inds]
            samples = self.concurrent_map_shadow(samples)

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
        batched_x = np.zeros([sequence_length,batch_size,self.train_x.shape[1],self.train_x.shape[2],self.train_x.shape[3]])
        batched_y = np.zeros([sequence_length,batch_size,1])

        # Fill each item of the batch buffer
        for i in range(batch_size):
            # Sample uniformly in training train
            uniform_sample = np.random.randint(0,self.train_y.shape[0]-sequence_length)

            # Right now we don't know if the sequence is valid (does not contain a frameskip)
            is_sample_valid = False
            while(not is_sample_valid):
                # Cut out sequence of length "sequence_length"
                sequence_indices = np.arange(uniform_sample,uniform_sample+sequence_length)

                # Timestamps in seconds
                timestamps = self.train_ts[sequence_indices]
                ts_diffs = timestamps[1:]-timestamps[:-1]

                if(np.max(ts_diffs) >= 5.0):
                    # Reject sample if frameskip is more than 5 seconds
                    uniform_sample = np.random.randint(0,self.train_y.shape[0]-sequence_length)
                else:
                    is_sample_valid = True
                
            # Copy to batch buffer
            batched_x[:,i] = self.train_x[sequence_indices].astype(np.float32)/255.0
            batched_y[:,i,:] = self.train_y[sequence_indices]
            # Augment each frame of sequences independently
            batched_x[:,i] = self.concurrent_map_shadow(batched_x[:,i])
        return (batched_x,batched_y*self._curvature_scaling)

    ''' Iterates over all the data split '''
    def iterate_as_single_sequence(self,max_seq_len=16,do_test = False):
        data_x,data_y,data_ts = self.valid_x, self.valid_y, self.valid_ts
        if(do_test):
            data_x,data_y,data_ts = self.test_x, self.test_y, self.test_ts

        # Count in how much mini-sequnces we can split the data
        number_of_chunks = data_x.shape[0]//max_seq_len
        # Flag indicating if there was a frameskip
        frameskip = True
        for i in range(number_of_chunks):
            sequence_index = np.arange(i*max_seq_len,(i+1)*max_seq_len)

            if(max_seq_len == 1):
                ts_diffs = data_ts[i]-data_ts[i-1]
                if(ts_diffs >= 5.0):
                    frameskip = True
            else:
                timestamps = data_ts[sequence_index]
                ts_diffs = timestamps[1:]-timestamps[:-1]

                # Remove frameskips
                if(np.max(ts_diffs) >= 5.0):
                    frameskip = True
                    continue

            y = data_y[sequence_index]
            x = data_x[sequence_index]

            yield (x,y*self._curvature_scaling,frameskip)
            frameskip = False


if __name__ == "__main__":
    root_path = "../cache_passive"
    passive_files = sorted([d for d in os.listdir(root_path) if d.endswith(".h5")])
    total_size = 0
    for p in passive_files:
        h5f = h5py.File(os.path.join(root_path, p),'r')
        print("Loading: ",str(p))
        y_raw = np.array(h5f['inverse_r'])
        total_size += y_raw.shape[0]

    print("Total frames: {:d}".format(total_size))