import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment to hide tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Uncomment to hide tensorflow logs
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from models.cnn_model import CnnModel
from models.e2e_worm_pilot import End2EndWormPilot
from models.e2e_lstm import End2EndLSTMPilot
from models.e2e_rnn import UniversalRNNPilot
from perspective_transformation import crop_all
import h5py
import sys
import time
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def writeline(f,id,arr):
    f.write("{:d}".format(id))
    for i in range(arr.shape[0]):
        f.write(";{:f}".format(arr[i]))
    f.write("\n")
    f.flush()

training_files =[

    # A lot of snow (afternoon):
    "../cache_active_right/feb/20190206-214345_blue_prius_devens_rightside.h5", # 5321 images
    # A lot of snow:
    "../cache_active_right/feb/20190206-213411_blue_prius_devens_rightside.h5", # 6208 images

    # dawn in January
    "../cache_active_right/feb/20190102-163800_blue_prius_devens_right_side.h5", # 4154 images

    # towards sun, melting snow
    "../cache_active_right/feb/20190206-212620_blue_prius_devens_rightside.h5", # 1622 images
    "../cache_active_right/feb/20190206-231418_blue_prius_devens_rightside.h5", # 5544 images

    # with snow
    "../cache_active_right/feb/20190205-103331_blue_prius_devens_rightside.h5", # 3601 images
    # "../cache_active_right/feb/20190205-144223_blue_prius_devens_rightside.h5", # 10093 images

    # fall, leaves on the road
    "../cache_active_right/nov/20181114-133340_blue_prius_devens_right_side.h5", # 4072 images

    # Summer
    "../cache_active_right/july/20180707-171536_blue_prius_devens_right_side.h5", # 19120 images
]
validation_files = [
    # Towards sun
    "../cache_active_right/feb/20190206-232946_blue_prius_devens_rightside.h5", # 2103 images
    # with snow
    "../cache_active_right/feb/20190205-102931_blue_prius_devens_rightside.h5", # 4162 images
    # Summer
    "../cache_active_right/july/20180714-171209_blue_prius_devens_right_side.h5", # 6017 images
]

# Parse arugments
parser = argparse.ArgumentParser(description='Train and test model with passive test data')
parser.add_argument('--debug',action='store_true',help='Debug Flag')
parser.add_argument('--model',  default='wm', help='Type of model. Options: cnn, e2e_wm and e2e_lstm')
parser.add_argument('--video',  default='none')
parser.add_argument('--saliency',  action="store_true" )
parser.add_argument('--clean',  action="store_true" ) # No noise will be applied
parser.add_argument('--replay_file',  default='None', help='Either a h5 or npy file')

# Example
# python3 replay_internal_states.py --saliency --video images --model wm --replay_file ../ros_analysis/camera_devens_feb_16/feb_16_wm_10_l_3_2019-02-06-22-26-21.npz
# python3 replay_internal_states.py --saliency --video images --model lstm --replay_file ../ros_analysis/camera_devens_feb_16/feb_16_lstm_forget1_3_2019-02-06-22-47-19.npz
# python3 replay_internal_states.py --saliency --video images --model cnn --replay_file ../ros_analysis/camera_devens_feb_16/feb_16_cnn_10_3_2019-02-06-22-36-29.npz


# python3 replay_internal_states.py --saliency --video images --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var0_2019-07-31-10-35-56.npz
# python3 replay_internal_states.py --saliency --video images --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test1_var0_2019-07-31-10-46-42.npz
# python3 replay_internal_states.py --saliency --video images --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test1_var0_2019-07-31-10-57-00.npz

# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test1_var01_2019-07-31-11-31-54.npz
# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test1_var0_2019-07-31-10-46-42.npz
# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test1_var02_2019-07-31-13-12-56.npz
# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test1_var03_2019-07-31-13-56-48.npz
# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test2_var0_2019-07-31-14-33-42.npz
# python3 replay_internal_states.py --model cnn --replay_file ../ros_analysis/camera_devens_jul31/cnn_test3_var0_2019-07-31-15-05-09.npz

# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test1_var01_2019-07-31-11-41-16.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test1_var0_2019-07-31-10-57-00.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test1_var02_2019-07-31-13-22-32.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test1_var03_2019-07-31-14-06-46.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test2_var0_2019-07-31-14-43-14.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test3_var0_2019-07-31-15-14-21.npz
# python3 replay_internal_states.py --model lstm --replay_file ../ros_analysis/camera_devens_jul31/lstm_test3_var0_2019-07-31-15-14-34.npz


# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var01_2019-07-31-11-21-42.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var0_2019-07-31-10-35-56.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var02_2019-07-31-12-56-38.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var025_2019-07-31-13-36-42.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var025_2019-07-31-13-38-42.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test1_var03_2019-07-31-13-46-58.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test2_var0_2019-07-31-14-24-17.npz
# python3 replay_internal_states.py --model wm --replay_file ../ros_analysis/camera_devens_jul31/wm_test3_var0_2019-07-31-14-54-53.npz


def plot_map(saliency_map,filename,norm=False):
    sns.set()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,1.17)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if(norm):
        # saliency_map -= np.min(saliency_map)
        # saliency_map /= (np.max(saliency_map)+0.00001)

        saliency_map -= np.mean(saliency_map)
        saliency_map /= (np.std(saliency_map)+0.00001)
        saliency_map = np.clip(saliency_map,-2,2)
        sns.heatmap(saliency_map,cbar=False,cmap=sns.cubehelix_palette(start=.5, rot=-.75,reverse=True))
        # sns.heatmap(saliency_map,vmin=0,vmax=1,cbar=False,cmap=sns.cubehelix_palette(start=.5, rot=-.75,reverse=True))
    else:
        # sns.heatmap(saliency_map,vmin=0,vmax=1,cbar=False,cmap=sns.cubehelix_palette(start=.5, rot=-.75,reverse=True))
        # sns.heatmap(saliency_map,vmin=0,vmax=1,cbar=False)
        
        saliency_map -= np.mean(saliency_map)
        saliency_map /= (np.std(saliency_map)+0.00001)
        saliency_map = np.clip(saliency_map,-3,3)

        quartile = np.percentile(saliency_map,10)
        saliency_map = np.clip(saliency_map,quartile,None)

        sns.heatmap(saliency_map,cbar=False)
    plt.axis('off')
    # plt.tight_layout()

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(filename, bbox_inches=extent, pad_inches=0)
    plt.close()

# Parse arguments and make sure a valid model has been selected
args = parser.parse_args()
if(not args.model in ['wm','lstm','cnn','ctrnn']):
    raise ValueError('Unknown model type: '+str(args.model))

wm_solver_unfolds = 6

if(args.model == 'wm'):
    model = End2EndWormPilot(
        wm_size = "mk2",
        conv_grad_scaling = 1.0,
        learning_rate=1.0,
        curve_factor = 1.0,
        # ode_solver_unfolds = wm_solver_unfolds,
    )
elif(args.model == 'lstm'):
    model = End2EndLSTMPilot(
        lstm_size = 64,
        conv_grad_scaling = 1.0,
        learning_rate=1.0,
        curve_factor = 1.0,
        clip_value = 10,
        forget_bias = 1.0,
    )
elif(args.model == 'cnn'):
    model = CnnModel(
        learning_rate = 1.0,
        curve_factor = 1.0,
        drf = 0.0,
        dr1 = 0.0,
        dr2 = 0.0
    )
elif(args.model == "ctrnn"):
    model = UniversalRNNPilot(
        num_units = 64,
        conv_grad_scaling = 1.0,
        learning_rate = 1.0,
        curve_factor = 1.0,
        clip_value = 10,
        rnn_type = "ctrnn",
        ctrnn_global_feedback = True,
    )
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.share_sess(sess)

noise_level = 0.0
if("var01_" in args.replay_file):
    noise_level = 0.1
elif("var02_" in args.replay_file):
    noise_level = 0.2
elif("var025_" in args.replay_file):
    noise_level = 0.25
elif("var03_" in args.replay_file):
    noise_level = 0.3

checkpoint_path = {
    "wm": "final_models/wm_ep_55",
    "lstm": "final_models/lstm_ep_12",
    "cnn": "final_models/cnn_ep_12",
    "ctrnn": "final_models/ctrnn ep_034",
}
model.restore_from_checkpoint(checkpoint_path[args.model])

if(not os.path.isfile(args.replay_file)):
    raise ValueError("File '{}' not found".format(args.replay_file))

trimmed_filename = os.path.basename(os.path.splitext(args.replay_file)[0])
if(args.clean):
    trimmed_filename += "_clean"
    noise_level = 0.0
    
export_dir = os.path.join("exported_replays",trimmed_filename)

if(not os.path.exists(export_dir)):
    os.makedirs(export_dir)

if(args.replay_file.endswith(".h5")):
    h5f = h5py.File(args.replay_file,'r')
    # Convert int64 array to uint8 array, to reduce memory footprint
    cam_raw = np.array(h5f['camera_front'])
    camera = crop_all(cam_raw)

    # Cache in memory
    timestamp = np.array(h5f['timestamp'],dtype=np.float64)
    ts_diffs = timestamp[1:]-timestamp[:-1]
    reset_mask = ts_diffs > 1.0

elif(args.replay_file.endswith(".npz")):
    arr = np.load(args.replay_file)
    camera = crop_all(arr["camera"])
    auto_mode = arr["auto"]
    reset_mask = np.logical_not(auto_mode)
    mask_1 = auto_mode[:-1]
    mask_2 = np.logical_not(auto_mode[1:])
    crashes = np.logical_and(mask_1, mask_2)
    crash_indices = np.arange(crashes.shape[0])[crashes]
    argmax_auto = np.argmax(auto_mode)
    print("argmax_auto: ",str(argmax_auto))
    with open(os.path.join(export_dir,"argmax_auto.csv"),"w") as f:
        f.write("{:d}".format(argmax_auto))

    print("Crash indices: ",str(crash_indices))

    np.savetxt(os.path.join(export_dir,"crash_indices.csv"),crash_indices)

if(args.model == "lstm"):
    lstm_w,lstm_b = sess.run([model.fused_cell._kernel,model.fused_cell._bias])
    input_w,new_w,forget_w,output_w = np.split(lstm_w,indices_or_sections=4,axis=1)
    input_b,new_b,forget_b,output_b = np.split(lstm_b,indices_or_sections=4,axis=0)
    # Forget bias
    forget_b += 1
    np.savetxt("final_models/lstm_params/input_w.csv",input_w,delimiter=";")
    np.savetxt("final_models/lstm_params/input_b.csv",input_b,delimiter=";")
    np.savetxt("final_models/lstm_params/new_w.csv",new_w,delimiter=";")
    np.savetxt("final_models/lstm_params/new_b.csv",new_b,delimiter=";")
    np.savetxt("final_models/lstm_params/forget_w.csv",forget_w,delimiter=";")
    np.savetxt("final_models/lstm_params/forget_b.csv",forget_b,delimiter=";")
    np.savetxt("final_models/lstm_params/output_w.csv",output_w,delimiter=";")
    np.savetxt("final_models/lstm_params/output_b.csv",output_b,delimiter=";")
elif(args.model == "ctrnn"):
    rnn_w,rnn_b = sess.run([model.fused_cell.W,model.fused_cell.b])
    input_part = rnn_w[:32,:]
    rec_part = rnn_w[32:,:]
    assert rec_part.shape[0]==rec_part.shape[1], "Recurrent weight should be symmetric"
    np.savetxt("final_models/ctrnn_params/w_recurrent.csv",rec_part,delimiter=";")
    np.savetxt("final_models/ctrnn_params/w_input.csv",input_part,delimiter=";")
    np.savetxt("final_models/ctrnn_params/b.csv",rnn_b,delimiter=";")

sensory_file = open(os.path.join(export_dir,"sensory.csv"),'w')
if(args.model == "wm" or args.model == "ctrnn"):
    state_file = open(os.path.join(export_dir,"neurons.csv"),'w')
elif(args.model == "lstm"):
    state_c_file = open(os.path.join(export_dir,"state_c.csv"),'w')
    state_h_file = open(os.path.join(export_dir,"state_h.csv"),'w')
output_file = open(os.path.join(export_dir,"output.csv"),'w')


unfolds = 1
if(args.model == "wm" and wm_solver_unfolds == 1):
    unfolds = 6


if(args.video == "mp4"):
    video_file = os.path.join(export_dir,"video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (camera.shape[2],camera.shape[1]))
    # you can change the default encoder using a four_cc
    # string, but not all of them work!
    # sv.encoder('H264')
    # sv.encoder('MP4V')
    print("Video file: ",str(video_file))
elif(args.video == "images"):
    video_dir = os.path.join(export_dir,"frames")
    if(not os.path.exists(video_dir)):
        os.makedirs(video_dir)
    if(args.saliency):
        saliency_dir = os.path.join(export_dir,"saliency_map")
        if(not os.path.exists(saliency_dir)):
            os.makedirs(saliency_dir)
        saliency_aux = os.path.join(export_dir,"saliency_aux")
        if(not os.path.exists(saliency_aux)):
            os.makedirs(saliency_aux)

has_started = False
rnn_state = None
for i in tqdm(range(camera.shape[0]-1)):

    # Reset RNN and RNG
    if(reset_mask[i] or i == 0):
        if(args.model == "wm"):
            rnn_state = np.zeros([model.wm.state_size])
        elif(args.model == "ctrnn"):
            rnn_state = model.zero_state(1)
        elif(args.model == "lstm"):
            rnn_state = model.zero_state(1)
        rng = np.random.RandomState(12345)

    img = camera[i]
    assert np.max(img) > 0
    # print("image max: ",str(np.max(img)))
    if(noise_level > 0.0):
        img = img/255.0
        img += rng.normal(loc=0,scale=noise_level,size=img.shape)
        img = np.clip(img,0,1)
        img *= 255
        img = img.astype(np.uint8)

    for u in range(unfolds):
        sensory_neurons, output, next_state = model.replay_internal_state(img,rnn_state)
        writeline(sensory_file,i,sensory_neurons)
        writeline(output_file,i,output)
        if(args.model == "wm" or args.model == "ctrnn"):
            writeline(state_file,i,rnn_state.flatten())
        elif(args.model == "lstm"):
            writeline(state_c_file,i,rnn_state.c.flatten())
            writeline(state_h_file,i,rnn_state.h.flatten())

        rnn_state = next_state
    if(args.video == "mp4"):
        cv2.putText(img, "{:05d}".format(i),(5,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        out.write(img)
    elif(args.video == "images") or args.video == "frames":
        if(args.saliency):
            saliency_map,aux_list = model.get_saliency_map(img)
            
            plot_map(saliency_map,os.path.join(saliency_dir,"frame_{:05d}.png".format(i)))

            for aux_id, aux_map in aux_list:
                plot_map(aux_map,os.path.join(saliency_aux,"frame_{:05d}_{}.png".format(i,aux_id)),norm=True)
            # if(i > 10):
            #     import sys
            #     sys.exit(1)
        cv2.imwrite(os.path.join(video_dir,"frame_{:05d}.jpg".format(i)),img)

if(args.video == "mp4"):
    out.release()

sensory_file.close()
output_file.close()
if(args.model == "wm" or args.model == "ctrnn"):
    state_file.close()
elif(args.model == "lstm"):
    state_c_file.close()
    state_h_file.close()