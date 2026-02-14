import numpy as np
import h5py
import tensorflow as tf
import os
import argparse

root_path = "../pretrained_passive_models"
passive_dirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

parser = argparse.ArgumentParser(description='List passive results')
parser.add_argument('--verbose', action="store_true")
parser.add_argument('--tex', action="store_true")
args = parser.parse_args()

for p in passive_dirs:
    all_results = []
    res_ids = []
    for i in range(10):
        res_file = os.path.join(root_path,p,"experiment_{:d}".format(i),"final.csv")
        # print('res path: ',str(res_file))
        if(os.path.isfile(res_file)):
            res = np.loadtxt(res_file,skiprows=1,delimiter=";")
            all_results.append(res)
            res_ids.append(i)
        
    if(len(all_results) == 0):
        print("Continue {} (only found {} results)".format(p,len(all_results)))
        continue

    all_results = np.stack(all_results)
    mean_result = np.mean(all_results,axis=0)
    std_result = np.std(all_results,axis=0)
    
    if(args.verbose):
        print("All results of model '{}'".format(p))
        for j in range(all_results.shape[0]):
            print("   Exp id [{}] valid loss: {:0.2f} mae: {:0.2}".format(res_ids[j],all_results[j,3],all_results[j,4]))

    if(args.tex):
        print("Model '{}' (results: {:}), train loss: {:0.2f}, mae: {:0.2f}, Valid loss: {:0.2f} mae: {:0.2} Test loss {:0.2f}, mae {:0.2f}".format(
            p,
            all_results.shape[0],
            mean_result[1],
            mean_result[2],
            mean_result[3],
            mean_result[4],
            mean_result[5],
            mean_result[6],
        ))
    else:
        print("{} & {:0.2f} $\\pm$ {:0.2f} & {:0.2f} $\\pm$ {:0.2f} &  {:0.2f} $\\pm$ {:0.2f} \\\\ ".format(
            p,
            mean_result[1],
            std_result[1],
            mean_result[5],
            std_result[5],
            mean_result[6],
            std_result[6],
        ))