"""
Summary:  Audio Set classification for ICASSP 2018 paper
Author:   Qiuqiang Kong, Yong Xu
Created:  2017.11.02

Summary:  Audio Set classification for Eusipco 2018 paper
Author:   Changsong Yu
Modified:  2018.02.21

"""
import os
import numpy as np
import h5py
import sys
import argparse
import time
import logging
import pickle as cPickle
from sklearn import metrics
#import theano
#import theano.tensor as T
os.environ['KERAS_BACKEND'] = "tensorflow"
import prepare_data as pp_data
from data_generator import RatioDataGenerator
from keras.models import Model
from keras.layers.core import *
from keras.layers import Input, Concatenate, BatchNormalization
from keras.callbacks import  Callback
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)
# Evaluate stadef eval(md, x, y, out_dir, out_probs_dir, iter_=iter_):
def eval(md, x, y, out_dir, out_probs_dir, iter_):
    
    # Predict
    t1 = time.time()
    (n_clips, n_time, n_freq) = x.shape
    (x, y) = pp_data.transform_data(x, y)
    prob = md.predict(x)
    prob = prob.astype(np.float32)
    
    if out_dir:
        pp_data.create_folder(out_dir)
        #out_prob_path = os.path.join(out_probs_dir, "prob_%d_iters.p" %iter_)
    # Dump predicted probabilites for future average
    if out_probs_dir:
        pp_data.create_folder(out_probs_dir)
        out_prob_path = os.path.join(out_probs_dir, "prob_%d_iters.p" %iter_)
        cPickle.dump(prob, open(out_prob_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Compute and dump stats
    n_out = y.shape[1]
    stats = []
    t1 = time.time()
    for k in range(n_out):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], prob[:, k])
        avg_precision = metrics.average_precision_score(y[:, k], prob[:, k], average=None)
        (fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], prob[:, k])
        auc = metrics.roc_auc_score(y[:, k], prob[:, k], average=None)
        #eer = pp_data.eer(prob[:, k], y[:, k])
        skip = 1000
        dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision, 
                'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
        
        stats.append(dict)
    logging.info("Callback time: %s" % (time.time() - t1,))
    
    dump_path = os.path.join(out_dir, "md%d_iters.p" % iter_)
    cPickle.dump(stats, open(dump_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    logging.info("mAP: %f" % np.mean([e['AP'] for e in stats]))

# Attention Lambda function
def _attention(inputs, **kwargs):
    [cla, att] = inputs
    
    _eps = 1e-7
    att = K.clip(att, _eps, 1. - _eps)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]
    #print("shape of normalized:", normalized_att.shape)
    return K.sum(cla * normalized_att, axis=1)

def _att_output_shape(input_shape):
    #print("shape of input_shape:", input_shape)
    return tuple([input_shape[0][0], input_shape[0][2]])
    
# Train the model
def train(args):
    cpickle_dir = args.cpickle_dir
    workspace = args.workspace
    
    # Path of hdf5 data
    bal_train_hdf5_path = os.path.join(cpickle_dir, "bal_train.h5")
    unbal_train_hdf5_path = os.path.join(cpickle_dir, "unbal_train.h5")
    eval_hdf5_path = os.path.join(cpickle_dir, "eval.h5")
    
    # Load data
    t1 = time.time()
    (tr_x1, tr_y1, tr_id_list1) = pp_data.load_data(bal_train_hdf5_path)
    (tr_x2, tr_y2, tr_id_list2) = pp_data.load_data(unbal_train_hdf5_path)    
    print(tr_x1.shape)
    print(tr_x2.shape)
    tr_x = np.concatenate((tr_x1, tr_x2))
    tr_y = np.concatenate((tr_y1, tr_y2))
    tr_id_list = tr_id_list1 + tr_id_list2

    (te_x, te_y, te_id_list) = pp_data.load_data(eval_hdf5_path)
    logging.info("Loading data time: %s s" % (time.time() - t1))
    
    logging.info(tr_x1.shape, tr_x2.shape)
    logging.info("tr_x.shape: %s" % (tr_x.shape,))
    
    (_, n_time, n_freq) = tr_x.shape
    
    # Build model
    n_hid = 600
    n_out = tr_y.shape[1]
    
    lay_in = Input(shape=(n_time, n_freq))
    a_0 = BatchNormalization()(lay_in)
    a_1 = Dense(n_hid, kernel_regularizer=regularizers.l2(0.001))(a_0)
    a_1 = BatchNormalization()(a_1)
    a_1 = Activation('relu')(a_1)
    a_1 = Dropout(rate=0.4)(a_1)
    a_2 = Dense(n_hid, kernel_regularizer=regularizers.l2(0.001))(a_1)
    a_2 = BatchNormalization()(a_2)
    a_2 = Activation('relu')(a_2)
    a_2 = Dropout(rate=0.4)(a_2)
    a_3 = Dense(n_hid, kernel_regularizer=regularizers.l2(0.001))(a_2)
    a_3 = BatchNormalization()(a_3)
    a_3 = Activation('relu')(a_3)
    a_3 = Dropout(rate=0.4)(a_3)
    cla_1 = Dense(n_out, name='cla_1')(a_3)
    cla_1 = BatchNormalization()(cla_1)
    cla_1 = Activation('sigmoid')(cla_1)
    att_1 = Dense(n_out, name='att_1')(a_3)
    att_1 = BatchNormalization()(att_1)
    att_1 = Activation('softmax')(att_1)
    
    # Attention
    lay_out_a = Lambda(_attention, output_shape=_att_output_shape)([cla_1, att_1])
    cla_2 = Dense(n_out, name='cla_2')(a_2)
    cla_2 = BatchNormalization()(cla_2)
    cla_2 = Activation('sigmoid')(cla_2)
    att_2 = Dense(n_out, name='att2')(a_2)
    att_2 = BatchNormalization()(att_2)
    att_2 = Activation('softmax')(att_2)
    
    lay_out_b = Lambda(_attention, output_shape=_att_output_shape)([cla_2, att_2])
    lay_out_c = Concatenate(axis=1)([lay_out_a, lay_out_b]) 

    #lay_out = Dense(n_out, activation='sigmoid', name='output')(lay_out_c)
    lay_out = Dense(n_out, name='output')(lay_out_c)
    lay_out = BatchNormalization()(lay_out)
    lay_out = Activation('sigmoid')(lay_out)
    # Compile model
    md = Model(inputs=lay_in, outputs=lay_out)
    md.summary()
    
    # Save model every several iterations
    call_freq = 1000
    dump_fd = os.path.join(workspace, "models", pp_data.get_filename(__file__))
    pp_data.create_folder(dump_fd)
   # save_model = SaveModel(dump_fd=dump_fd, call_freq=call_freq, type='iter', is_logging=True)
    
    # Callbacks function
    #callbacks = []#save_model]
    
    batch_size = 500
    tr_gen = RatioDataGenerator(batch_size=batch_size, type='train')
    
    # Optimization method
    optimizer = Adam(lr=args.lr)
    md.compile(loss='binary_crossentropy', 
               optimizer=optimizer)
               #callbacks=callbacks)
    
    # Train
    stat_dir = os.path.join(workspace, "stats", pp_data.get_filename(__file__))
    pp_data.create_folder(stat_dir)
    prob_dir = os.path.join(workspace, "probs", pp_data.get_filename(__file__))
    pp_data.create_folder(prob_dir)
    
    tr_time = time.time()
    iter_ = 1
    for (tr_batch_x, tr_batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        # Compute stats every several interations
        if iter_ % call_freq == 0:
            # Stats of evaluation dataset
            t1 = time.time()
            te_err = eval(md=md, x=te_x, y=te_y, 
                          out_dir=os.path.join(stat_dir, "test"), 
                          out_probs_dir=os.path.join(prob_dir, "test"), iter_=iter_)
            logging.info("Evaluate test time: %s" % (time.time() - t1,))
            
            # Stats of training dataset
            t1 = time.time()
            tr_bal_err = eval(md=md, x=tr_x1, y=tr_y1, 
                              out_dir=os.path.join(stat_dir, "train_bal"), 
                              out_probs_dir=None, iter_=iter_)
            logging.info("Evaluate tr_bal time: %s" % (time.time() - t1,))
        iter_ += 1
        # Update params
        (tr_batch_x, tr_batch_y) = pp_data.transform_data(tr_batch_x, tr_batch_y)
        md.train_on_batch(x=tr_batch_x, y=tr_batch_y) 
        # Stop training when maximum iteration achieves
        if iter_ == call_freq * 151:
            break


# Average predictions of different iterations and compute stats
def get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter):
    eval_hdf5_path = os.path.join(args.cpickle_dir, "eval.h5")
    workspace = args.workspace
    
    # Load ground truth
    (te_x, te_y, te_id_list) = pp_data.load_data(eval_hdf5_path)
    y = te_y
    
    # Average prediction probabilities of several iterations
    prob_dir = os.path.join(workspace, "probs", file_name, "test")
    names = os.listdir(prob_dir)
    
    probs = []
    iters = range(bgn_iter, fin_iter, interval_iter)
    for iter in iters:
        pickle_path = os.path.join(prob_dir, "prob_%d_iters.p" % iter)
        prob = cPickle.load(open(pickle_path, 'rb'))
        probs.append(prob)
    #print(len(probs))
    avg_prob = np.mean(np.array(probs), axis=0)

    # Compute stats
    t1 = time.time()
    n_out = y.shape[1]
    stats = []
    for k in range(n_out):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], avg_prob[:, k])
        avg_precision = metrics.average_precision_score(y[:, k], avg_prob[:, k], average=None)
        (fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], avg_prob[:, k])
        auc = metrics.roc_auc_score(y[:, k], avg_prob[:, k], average=None)
        #eer = pp_data.eer(avg_prob[:, k], y[:, k])
        
        skip = 1000
        dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision, 
                'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
        
        stats.append(dict)
    logging.info("Callback time: %s" % (time.time() - t1,))
    
    # Dump stats
    dump_path = os.path.join(workspace, "stats", pp_data.get_filename(__file__), "test", "avg_%d_%d_%d.p" % (bgn_iter, fin_iter, interval_iter))
    pp_data.create_folder(os.path.dirname(dump_path))
    cPickle.dump(stats, open(dump_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    #print(stats.shape)
    #for i, e in enumerate(stats):
    #  logging.info("%d. mAP: %f, auc: %f, d_prime: %f" % (i, e['AP'], e['auc'], pp_data.d_prime(e['auc'])))

    # Write out to log
    logging.info("bgn_iter, fin_iter, interval_iter: %d, %d, %d" % (bgn_iter, fin_iter, interval_iter))
    logging.info("mAP: %f" % np.mean([e['AP'] for e in stats]))
    auc = np.mean([e['auc'] for e in stats])
    logging.info("auc: %f" % auc)
    logging.info("d_prime: %f" % pp_data.d_prime(auc))
    
    
# Main
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--cpickle_dir', type=str)
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--lr', type=float, default=1e-3)
    
    parser_get_avg_stats = subparsers.add_parser('get_avg_stats')
    parser_get_avg_stats.add_argument('--cpickle_dir', type=str)
    parser_get_avg_stats.add_argument('--workspace')

    args = parser.parse_args()
    
    # Logs
    logs_dir = os.path.join(args.workspace, "logs", pp_data.get_filename(__file__))
    pp_data.create_folder(logs_dir)
    logging = pp_data.create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    if args.mode == "train":
        train(args)
    elif args.mode == 'get_avg_stats':
        file_name=pp_data.get_filename(__file__)
        #for k in range(20000, 30000, 1000):
        bgn_iter, fin_iter, interval_iter = 1000, 55001, 1000
        get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter)
    else:
        raise Exception("Error!")
