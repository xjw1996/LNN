import numpy as np
import tensorflow as tf
from models.cnn_model import CnnModel
from models.e2e_worm_pilot import End2EndWormPilot
from models.e2e_lstm import End2EndLSTMPilot
from models.lr_model import LinearRegressionModel
from models.e2e_rnn import UniversalRNNPilot
from passive_test_data_provider import PassiveTestDataProvider
import sys
import time
import os
import argparse

def evaluate_on_validation(model,data_provider,max_seq_len,do_test=False):
    losses = []
    abs_erros = []
    rnn_state = None
    for x,y,frameskip in data_provider.iterate_as_single_sequence(max_seq_len,do_test):
        # If there was a frameskip we need to reset the RNN
        if(frameskip):
            rnn_state = None

        loss,mae,rnn_state = model.evaluate_single_sequence(x,y,rnn_state)
        losses.append(loss)
        abs_erros.append(mae)

    return (np.mean(losses),np.mean(abs_erros))

def time2str(elapsed):
    sec = int(elapsed)
    mins = sec // 60
    sec = sec % 60
    return "{:02d}:{:02d}".format(mins,sec)

# Parse arugments
parser = argparse.ArgumentParser(description='Train and test model with passive test data')
parser.add_argument('--debug',action='store_true',help='Debug Flag')
parser.add_argument('--final',action='store_true',help='Evaluate on the test set')
parser.add_argument('--experiment_id',  type=int, default=-1, help='Experiment id [0,9]')

# Hyperparameters for training
parser.add_argument('--lr', default=0.0005, type=float,help='Learning Rate (default: 0.0005)')
parser.add_argument('--batch_size',  type=int, default=20, help='Batch size')
parser.add_argument('--conv_grad_scaling', default=1.0, type=float,help='Scaling factor of the convolution layer gradients (RNNs only)')

parser.add_argument('--epochs',  type=int, default=50, help='Number of training epochs')
parser.add_argument('--curve_factor', default=0.1, type=float,help='Factor in the exponential term of the sample weighting (0 means no weighting at all)')
parser.add_argument('--shadow_gamma', default=1.0, type=float,help='Maximum value of gamma distortion of the shadow augmentation')
parser.add_argument('--darkening_ratio', default=0.66, type=float,help='Ratio of show many shadows are darkened vs lightened')

# Specification of the model (which model, where to store it, ...)
parser.add_argument('--new',action='store_true',help='Overwrites existing base_path')
parser.add_argument('--base_path',  default='session', help='Base path to store the sessions')
parser.add_argument('--model',  default='None', help='Type of model. Options: cnn, e2e_wm and e2e_lstm')

# RNN specific parameters
parser.add_argument('--seq_len',  type=int, default=16, help='Sequence length')
parser.add_argument('--sparsity', default=-1.0, type=float,help='Sparsity level 0 mean fully connected, 1 means empty network')

# Model specific parameters
# CNN:
parser.add_argument('--drf', default=0.5, type=float,help='Dropout keep_prob for flattened layer (default: 0.5)')
parser.add_argument('--dr1', default=0.5, type=float,help='Dropout keep_prob for first FC layer  (default: 0.5)')
parser.add_argument('--dr2', default=0.7, type=float,help='Dropout keep_prob for second FC layer (default: 0.7)')
# LSMT:
parser.add_argument('--lstm_size',  type=int, default=64, help='Number of LSTM cells')
parser.add_argument('--lstm_clip',  type=float, default=-1, help='Clip LSTM memory values')
parser.add_argument('--lstm_forget_bias',  type=float, default=1, help='Forget bias of LSTM cell')
# Wormnet
parser.add_argument('--wm_size',  default='small', help='Use denser wormnet architecture')
# CT-RNN
parser.add_argument('--ctfeedback',action='store_true',help='Enable global feedback for the CT-RNN')

# Logging period (evaluates on validation data, prints metrics and creates a checkpoint)
parser.add_argument('--log_period',  type=int, default=5, help='Log period for evaluating validation performance')

# Parse arguments and make sure a valid model has been selected
args = parser.parse_args()
if(not args.model in ["cnn","wm","lstm", "wm","gru","vanilla","ctgru","ctrnn","tf_gru"]):
    raise ValueError('Unknown model type: '+str(args.model))
if(args.experiment_id < 0 or args.experiment_id > 9):
    raise ValueError("Invalid experiment id: {}".format(args.experiment_id))

# Base directory to store training history, checkpoints, etc.
base_path = os.path.join("passive_sessions","{}_{}".format(args.model,args.base_path),"experiment_{:d}".format(args.experiment_id))
training_history_log = os.path.join(base_path,"train_log.csv")
final_evaluation_log = os.path.join(base_path,"final.csv")

if(os.path.exists(base_path)):
    if(args.new == False):
        raise ValueError('Session directory already exist, but neither --restore nor --new command line options were specified!')
else:
    os.makedirs(base_path)

# Create header line of training curve csv file
with open(training_history_log, 'w') as f:
    f.write("epoch; train_loss; train_mae; valid_loss; valid_mae\n")

# Load data
print("Loading data ...")
data_provider = PassiveTestDataProvider(
    h5_files_directory="../cache_passive",
    experiment_id=args.experiment_id,
    shadow_max_gamma=args.shadow_gamma,
    shadow_darkening_ratio=args.darkening_ratio,
    debug_flag=args.debug,
)
data_provider.summary()

is_rnn_model = args.model in ["lstm", "wm","gru","vanilla","ctgru","ctrnn","tf_gru"]

if(args.model == 'cnn'):
    model = CnnModel(
        learning_rate = args.lr,
        curve_factor = args.curve_factor,
        drf = args.drf,
        dr1 = args.dr1,
        dr2 = args.dr2
    )
elif(args.model == 'wm'):
    model = End2EndWormPilot(
        wm_size = args.wm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor = args.curve_factor,
    )
elif(args.model == 'lstm'):
    model = End2EndLSTMPilot(
        lstm_size = args.lstm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor = args.curve_factor,
        clip_value = args.lstm_clip,
        forget_bias = args.lstm_forget_bias,
        sparsity_level=args.sparsity,
    )
elif(args.model in ['gru','vanilla','ctgru','ctrnn','tf_gru']):
    model = UniversalRNNPilot(
        num_units = args.lstm_size,
        conv_grad_scaling = args.conv_grad_scaling,
        learning_rate = args.lr,
        curve_factor = args.curve_factor,
        clip_value = args.lstm_clip,
        rnn_type = args.model,
        ctrnn_global_feedback = args.ctfeedback,
        sparsity_level=args.sparsity,
    )

sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.share_sess(sess)
# print("All TF vars:")
# tf_vars = tf.trainable_variables()
# for v in tf_vars:
#     print(" Variable {}".format(str(v)))

best_valid_loss = np.PINF
best_valid_epoch = 0
# Start training
training_start = time.time()
print('Entering training loop')
for epoch in range(args.epochs):
    new_best_epoch = False
    # Evaluate test metric before the training epoch
    if epoch % args.log_period == 0 or epoch == args.epochs-1:
        valid_loss,valid_mae = evaluate_on_validation(model,data_provider,max_seq_len=16)
        if(valid_loss < best_valid_loss):
            best_valid_epoch = epoch
            best_valid_loss = valid_loss
            best_valid_mae = valid_mae
            new_best_epoch = True
            model.create_checkpoint(os.path.join(base_path,"session"))

    train_losses = []
    train_abs_erros = []

    if(is_rnn_model):
        # Train with sequences (for wormnet and LSTM)
        for e in range(data_provider.count_training_set(args.batch_size,args.seq_len)):
            batch_x,batch_y = data_provider.create_sequnced_batch(args.batch_size,args.seq_len)
            loss, abs_err = model.train_iter(batch_x,batch_y)
            train_losses.append(loss)
            train_abs_erros.append(abs_err)
    else:
        # Train with shuffeled images and dropout (for CNN)
        for batch_x,batch_y in data_provider.iterate_shuffled_train(args.batch_size):
            # Train with dropout
            loss, abs_err = model.train_iter(batch_x,batch_y)
            train_losses.append(loss)
            train_abs_erros.append(abs_err)

    # Report training and test loss and error after the epochs
    if epoch % args.log_period == 0 or epoch == args.epochs-1:
        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_abs_erros)
        # If the last epoch was a new best, we need to record the training metrics
        if(new_best_epoch):
            best_train_loss = train_loss
            best_train_mae = train_mae
        with open(training_history_log, 'a') as f:
            f.write("{}; {:0.3f}; {:0.3f}; {:0.3f}; {:0.3f}\n".format(
                epoch,
                train_loss,train_mae,
                valid_loss,valid_mae
            ))
        # Also print the same line
        print('Metrics after {} epochs, train loss: {:.2f}, train mae: {:.2f}, valid loss: {:.2f}, valid mae: {:.2f}'.format(
            epoch,train_loss,train_mae,valid_loss,valid_mae
        ))

training_time_per_epoch = (time.time() - training_start)/args.epochs
print("Mean training time per epoch: {}".format(time2str(training_time_per_epoch)))

print("Best epoch: {:}, train_loss: {:0.2f}, train_mae: {:0.2f}, valid_loss: {:0.2f}, valid_mae: {:0.2f}".format(
    best_valid_epoch,
    best_train_loss,best_train_mae,
    best_valid_loss,best_valid_mae
))
if(args.final):
    # Restore the checkpoint with the best validation loss to evaluate on the test set
    model.restore_from_checkpoint(os.path.join(base_path,"session"))
    test_loss,test_mae = evaluate_on_validation(model,data_provider,max_seq_len=16,do_test=True)
    print("Test loss: {:0.2f}, test mae: {:0.2f}".format(
        test_loss,test_mae
    ))

with open(final_evaluation_log, 'w') as f:
    f.write("best_epoch; train_loss; train_mae; valid_loss; valid_mae; test_loss; test_mae\n")
    f.write("{}; {:0.5f}; {:0.5f}; {:0.5f}; {:0.5f}".format(
        best_valid_epoch,
        best_train_loss,best_train_mae,
        best_valid_loss,best_valid_mae
    ))
    if(args.final):
        f.write("; {:0.5f}; {:0.5f}".format(
            test_loss,test_mae
        ))
