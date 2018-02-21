"""
Summary:  Audio Set classification for ICASSP 2018 paper
Author:   Qiuqiang Kong, Yong Xu
Created:  2017.11.02

Summary:  Audio Set classification for Eusipco 2018 paper
Author:   Changsong Yu
Modified:  2018.02.21

"""
import numpy as np
import os
import gzip
import h5py
import logging
from scipy import stats
from sklearn import metrics
import time
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
    
# Logging
def create_logging(log_dir, filemode):
    # Write out to file
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, "%05d.log" % i1)):
        i1 += 1
    log_path = os.path.join(log_dir, "%05d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging
    
def eer(pred, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, drop_intermediate=True)
    
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)  
    return EER

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc)*np.sqrt(2.0)
    return d_prime

# Load data    
def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = np.array(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.
    
def bool_to_float32(y):
    return np.float32(y)
    
def transform_data(x, y):
    x = uint8_to_float32(x)
    y = bool_to_float32(y)
    return x, y

### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
#        na_list = list(hf.get('na_list'))
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
 #       print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y#, na_list
  
  
def do_scale(x3d, scaler_path, verbose=1):
   """Do scale on the input sequence data. 
   
   Args:
     x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
     scaler_path: string, path of pre-calculated scaler. 
     verbose: integar, print flag. 
     
   Returns:
     Scaled input sequence data. 
   """
   t1 = time.time()
   scaler = pickle.load(open(scaler_path, 'rb'))
   (n_clips, n_time, n_freq) = x3d.shape
   x2d = x3d.reshape((n_clips * n_time, n_freq))
   x2d_scaled = scaler.transform(x2d)
   x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
   if verbose == 1:
       print("Scaling time: %s" % (time.time() - t1,))
   return x3d_scaled
