import numpy as np
import tensorflow as tf
from models.cnn_model import CnnModel
from models.e2e_worm_pilot import End2EndWormPilot
from models.e2e_lstm import End2EndLSTMPilot
from models.e2e_rnn import UniversalRNNPilot
from active_data_provider import ActiveDataProvider
import sys
import time
import os
import argparse

def evaluate_on_validation(model,data_provider,max_seq_len):
    losses = []
    abs_erros = []
    rnn_state = None
    for x,y,frameskip in data_provider.iterate_as_single_sequence(max_seq_len):
        # If there was a frameskip we need to reset the RNN
        if(frameskip):
            rnn_state = None

        loss,mae,rnn_state = model.evaluate_single_sequence(x,y,rnn_state)
        losses.append(loss)
        abs_erros.append(mae)

    return (np.mean(losses),np.mean(abs_erros))


# Parse arugments
parser = argparse.ArgumentParser(description='Train and test model with passive test data')
parser.add_argument('--debug',action='store_true',help='Debug Flag')

training_files =[
    "../cache_active_right/2019/20190628-094335_blue_prius_devens_rightside.h5", # 2,6G
    "../cache_active_right/2019/20190723-133449_blue_prius_devens_rightside.h5", # 2,6G
    "../cache_active_right/2019/20190723-161821_blue_prius_devens_rightside.h5", # 2,1G
    "../cache_active_right/2019/20190723-154501_blue_prius_devens_rightside.h5", # 1,9G 
]
validation_files = [
    "../cache_active_right/2019/20190628-150233_blue_prius_devens_rightside.h5", # 2,3G
    "../cache_active_right/2019/20190723-134708_blue_prius_devens_rightside.h5", # 911M 
]

# Hyperparameters for training
parser.add_argument('--lr', default=0.0005, type=float,help='Learning Rate (default: 0.0005)')
parser.add_argument('--batch_size',  type=int, default=20, help='Batch size')
parser.add_argument('--conv_grad_scaling', default=1.0, type=float,help='Scaling factor of the convolution layer gradients (RNNs only)')

parser.add_argument('--epochs',  type=int, default=200, help='Number of training epochs')
parser.add_argument('--curve_factor', default=0.0, type=float,help='Factor in the exponential term of the sample weighting (0 means no weighting at all)')
parser.add_argument('--shadow_gamma', default=0.0, type=float,help='Maximum value of gamma distortion of the shadow augmentation')
parser.add_argument('--darkening_ratio', default=0.66, type=float,help='Ratio of show many shadows are darkened vs lightened')

# Specification of the model (which model, where to store it, ...)
parser.add_argument('--new',action='store_true',help='Overwrites existing base_path')
parser.add_argument('--restore',action='store_true',help='Continues training an existing session')
parser.add_argument('--base_path',  default='session', help='Base path to store the sessions')
parser.add_argument('--model',  default='None', help='Type of model. Options: cnn, e2e_wm and e2e_lstm')

# RNN specific parameters
parser.add_argument('--seq_len',  type=int, default=16, help='Sequence length')

# Model specific parameters
# CNN:
parser.add_argument('--drf', default=0.5, type=float,help='Dropout keep_prob for flattened layer (default: 0.5)')
parser.add_argument('--dr1', default=0.5, type=float,help='Dropout keep_prob for first FC layer  (default: 0.5)')
parser.add_argument('--dr2', default=0.7, type=float,help='Dropout keep_prob for second FC layer (default: 0.7)')
# LSMT:
parser.add_argument('--lstm_size',  type=int, default=64, help='Number of LSTM cells')
parser.add_argument('--lstm_clip',  type=float, default=10, help='Clip LSTM memory values')
parser.add_argument('--lstm_forget_bias',  type=float, default=1.0, help='Forget bias of LSTM cell')
# Wormnet
parser.add_argument('--wm_size',  default='mk2', help='Use denser wormnet architecture')

# Logging period (evaluates on validation data, prints metrics and creates a checkpoint)
parser.add_argument('--log_period',  type=int, default=1, help='Log period for evaluating validation performance')

# Parse arguments and make sure a valid model has been selected
args = parser.parse_args()
if(not args.model in ['cnn','e2e_lstm','e2e_wm','e2e_ctrnn']):
    raise ValueError('Unknown model type: '+str(args.model))

# Base directory to store training history, checkpoints, etc.
base_path = os.path.join("active_sessions","{}_{}".format(args.model,args.base_path))
training_history_log = os.path.join(base_path,"train_{}_{}.csv".format(args.model,args.base_path))

if(os.path.exists(base_path)):
    if(args.restore):
        raise NotImplementedError("Continuing a session is currently not implemented")
    elif(args.new == False):
        raise ValueError('Session directory already exist, but neither --restore nor --new command line options were specified!')
else:
    os.makedirs(base_path)

# Create header line of training curve csv file
with open(training_history_log, 'w') as f:
    f.write("epoch; train_loss; train_mae; test_loss; test_mae\n")

# Load data
print("Loading data ...")
train_data_provider = ActiveDataProvider(
    h5_files = training_files,
    shadow_max_gamma = args.shadow_gamma,
    shadow_darkening_ratio = args.darkening_ratio,
    debug_flag=args.debug)
train_data_provider.summary(set_name="Training set")

valid_data_provider = ActiveDataProvider(
    h5_files = validation_files,
    debug_flag=args.debug)
valid_data_provider.summary(set_name="Validation set")

is_rnn_model = args.model in ["e2e_lstm", "e2e_wm","e2e_ctrnn"]

if(args.model == 'cnn'):
    model = CnnModel(
        learning_rate = args.lr,
        curve_factor = args.curve_factor,
        drf = args.drf,
        dr1 = args.dr1,
        dr2 = args.dr2
    )
elif(args.model == 'e2e_wm'):
    model = End2EndWormPilot(
        wm_size = args.wm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor = args.curve_factor,
    )
elif(args.model == 'e2e_lstm'):
    model = End2EndLSTMPilot(
        lstm_size = args.lstm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor = args.curve_factor,
        clip_value = args.lstm_clip,
        forget_bias = args.lstm_forget_bias,
    )
elif(args.model == "e2e_ctrnn"):
    model = UniversalRNNPilot(
        num_units = args.lstm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate = args.lr,
        curve_factor = args.curve_factor,
        clip_value = args.lstm_clip,
        rnn_type = "ctrnn",
        ctrnn_global_feedback = True,
    )


sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.share_sess(sess)

# Start training
print('Entering training loop')
for epoch in range(args.epochs):
    # Evaluate test metric before the training epoch
    if epoch % args.log_period == 0 or epoch == args.epochs-1:
        checkpoint_dir = os.path.join(base_path,"checkpoints","epoch_{:03d}".format(epoch))
        # Create new saver object
        model.create_checkpoint(checkpoint_dir)
        # The parameter seq_len does not necessarily need to be equal to the training seq_len 
        # it's just to speed up computation
        test_loss,test_mae = evaluate_on_validation(model,valid_data_provider,max_seq_len=32)

        if(args.model == "e2e_wm"):
            # If we are using a wormnet, we also export the parameters to a textfile
            dump_dir = os.path.join(base_path,"checkpoints","epoch_{:03d}".format(epoch),"wm_dump")
            model.wm.export_parameters(sess,dump_dir)
        elif(args.model == "e2e_lstm"):
            # Export parameters of forget gate
            lstm_w,lstm_b = sess.run([model.fused_cell._kernel,model.fused_cell._bias])
            _,_,forget_w,_ = np.split(lstm_w,indices_or_sections=4,axis=1)
            _,_,forget_b,_ = np.split(lstm_b,indices_or_sections=4,axis=0)
            np.savetxt(os.path.join(base_path,"checkpoints","epoch_{:03d}".format(epoch),"forget_bias.csv"),forget_b)
            np.savetxt(os.path.join(base_path,"checkpoints","epoch_{:03d}".format(epoch),"forget_w.csv"),forget_w)


    train_losses = []
    train_abs_erros = []

    if(is_rnn_model):
        # Train with sequences (for wormnet and LSTM)
        for e in range(train_data_provider.count_epoch_size(args.batch_size,args.seq_len)):
            batch_x,batch_y = train_data_provider.create_sequnced_batch(args.batch_size,args.seq_len)
            loss, abs_err = model.train_iter(batch_x,batch_y)
            train_losses.append(loss)
            train_abs_erros.append(abs_err)
    else:
        # Train with shuffeled images and dropout (for CNN)
        for batch_x,batch_y in train_data_provider.iterate_shuffled_train(args.batch_size):
            # Train with dropout
            loss, abs_err = model.train_iter(batch_x,batch_y)
            train_losses.append(loss)
            train_abs_erros.append(abs_err)

    # Report training and test loss and error after the epochs
    if epoch % args.log_period == 0 or epoch == args.epochs-1:
        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_abs_erros)
        with open(training_history_log, 'a') as f:
            f.write("{}; {}; {}; {}; {}\n".format(
                epoch,
                train_loss,train_mae,
                test_loss,test_mae
            ))
        # Also print the same line
        print('Metrics after {} epochs, train loss: {:.2f}, train mae: {:.2f}, test loss: {:.2f}, test mae: {:.2f}'.format(
            epoch,train_loss,train_mae,test_loss,test_mae
        ))

# Done training, now do the final evaluation
model.create_checkpoint(os.path.join(base_path,"checkpoints","final"))
test_loss,test_mae = evaluate_on_validation(model,valid_data_provider,max_seq_len=32)

# Add final line to training history
train_loss = np.mean(train_losses)
train_mae = np.mean(train_abs_erros)
with open(training_history_log, 'a') as f:
    f.write("{}; {}; {}; {}; {}\n".format(
        args.epochs,
        train_loss,train_mae,
        test_loss,test_mae
    ))

# Also print the same line
print('Metrics after all {} epochs, train loss: {:.2f}, train mae: {:.2f}, test loss: {:.2f}, test mae: {:.2f}'.format(
    args.epochs,train_loss,train_mae,test_loss,test_mae
))
