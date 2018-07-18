"""
input pipeline for google robotic data
"""

import numpy as np
import tensorflow as tf
import parameters as param
import os
import copy

#===data for optical flow version of Deep RBF network===
def DataForDRBF_OF(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;
    
    #===network's input file names===    
    x_dir=rootdir+'/OpticalFlows/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_filenames=copy.copy(x_filenames)
    #===network's output file names===
    
    #===error masks file names===
    e_dir=rootdir+'/ErrorMasks/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]    
    #===error masks file names===
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
            
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*im_h*im_w;y_size=(im_h,im_w)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch(x_train_param,y_train_param,
                                                       e_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch(x_test_param,y_test_param,
                                                       e_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for optical flow version of Deep RBF network===

#===data for optical flow version of Berkeley network===
def DataForBRN_OF(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;
    
    #===network's input file names===    
    x_dir=rootdir+'/OpticalFlows/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/DonwSampledOF/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]
    #===network's output file names===
    
    #===error masks file names===
    e_dir=rootdir+'/ErrorMasks/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]    
    #===error masks file names===
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
            
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*32*32;y_size=(32,32)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch(x_train_param,y_train_param,
                                                       e_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch(x_test_param,y_test_param,
                                                       e_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for optical flow version of Deep RBF network===

#===data for frame difference version of Deep RBF network===  
def DataForDRBF_ID(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;
    
    #===network's input file names===    
    x_dir=rootdir+'/diff_frames/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/pre_frames/video_'
    y_filenames=[]
    for vid in range(n_video):
        y_filenames+=np.arange(2,param.N_MOVE-3,3).size*\
                    [y_dir+'0'*(5-len(str(vid)))+str(vid)]
    #===network's output file names===
    
    #===pre frame file names===
    e_dir=rootdir+'/ErrorMasks_ID/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-3,3)]  
    #===error masks file names===
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
            
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*im_h*im_w;y_size=(im_h,im_w)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch(x_train_param,y_train_param,
                                                       e_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch(x_test_param,y_test_param,
                                                       e_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test 
#===data for frame difference version of Deep RBF network===  


#===data for State prediction network===
def DataForSPN(batch_size,p_train,method):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;
    
    #===network's input file names===    
    x_dir=rootdir+'/SPN_'+method+'_input/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/SPN_'+method+'_output/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]
    #===network's output file names===
    
    #===error masks file names===
    e_dir=rootdir+'/ErrorMasks/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]    
    #===error masks file names===
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
    
    if method=='BR_OF':
        x_nb=4*(3*param.N_command+32);
        x_size=(1,3*param.N_command+32)
        y_nb=4*32;y_size=(1,32)   
        e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    else:        
        x_nb=4*(3*param.N_command+param.N_RBF);
        x_size=(1,3*param.N_command+param.N_RBF)
        y_nb=4*param.N_RBF;y_size=(1,param.N_RBF)   
        e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch(x_train_param,y_train_param,
                                                       e_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch(x_test_param,y_test_param,
                                                       e_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test      
#===data for State prediction network===

#===data for State prediction network===
def DataForPCAN_OF(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;
        
    #===network's input file names===    
    x_dir=rootdir+'/PCAN_OF_input/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/PCAN_OF_output/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]
    #===network's output file names===
    
    #===error masks file names===
    e_dir=rootdir+'/ErrorMasks/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]    
    #===error masks file names===
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
            
    x_nb=4*(3*param.N_command+param.N_PCA);
    x_size=(1,3*param.N_command+param.N_PCA)
    y_nb=4*param.N_PCA;
    y_size=(1,param.N_PCA)   
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch(x_train_param,y_train_param,
                                                       e_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch(x_test_param,y_test_param,
                                                       e_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for State prediction network===

#===creat tf queue from filename lists===
def createQueuedShuffledBatch(x_param,y_param,e_param,
                              batch_size,shuffle):
    
    x_filenames,x_nb,x_shape=x_param        
    x_filename_queue=tf.train.string_input_producer(x_filenames,shuffle=False)    
    x_reader = tf.FixedLengthRecordReader(record_bytes=x_nb)   
    x_key, x_value = x_reader.read(x_filename_queue)
    x_record_bytes = tf.decode_raw(x_value, tf.float32)       
    x_im=tf.reshape(x_record_bytes,x_shape)
    
    y_filenames,y_nb,y_shape=y_param
    y_filename_queue=tf.train.string_input_producer(y_filenames,shuffle=False)    
    y_reader = tf.FixedLengthRecordReader(record_bytes=y_nb)   
    y_key, y_value = y_reader.read(y_filename_queue)
    y_record_bytes = tf.decode_raw(y_value, tf.float32)       
    y_im=tf.reshape(y_record_bytes,y_shape)
    
    e_filenames,e_nb,e_shape=e_param
    e_filename_queue=tf.train.string_input_producer(e_filenames,shuffle=False)    
    e_reader = tf.FixedLengthRecordReader(record_bytes=e_nb)   
    e_key, e_value = e_reader.read(e_filename_queue)
    e_record_bytes = tf.decode_raw(e_value, tf.float32)       
    e_im=tf.reshape(e_record_bytes,e_shape)
    
    if shuffle==1:    
        images = tf.train.shuffle_batch([x_im,y_im,e_im],
                                              batch_size,capacity=500,
                                              min_after_dequeue=100)
    else:
        images = tf.train.batch([x_im,y_im,e_im],
                                              batch_size,capacity=500)
    return images
#===creat tf queue from filename lists===



#===data for optical flow version of 1 step prediction network===
def DataForOSAP_OF(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;        
    
    #===joints position===
    a_dir=rootdir+'/JointPos/video_'
    a_filenames=[a_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===joints position===
    
    #===network's input file names===    
    x_dir=rootdir+'/OpticalFlows/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/OpticalFlows/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]
    #===network's output file names===
    
    #===error masks file names===
    e_dir=rootdir+'/ErrorMasks/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]    
    #===error masks file names===        
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
    a_train_filenames=list(np.asarray(a_filenames)[ind[0:n_train]])
    a_test_filenames=list(np.asarray(a_filenames)[ind[n_train:N]])       
    
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*im_h*im_w;y_size=(im_h,im_w)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    a_nb=4*42;a_size=(1,42)#each joint has 7 elements and we consider 6 of them
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)
    a_train_param=(a_train_filenames,a_nb,a_size)
    a_test_param=(a_test_filenames,a_nb,a_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch_OSAP(x_train_param,y_train_param,
                                                       e_train_param,
                                                       a_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch_OSAP(x_test_param,y_test_param,
                                                       e_test_param,
                                                       a_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for optical flow version of 1 step prediction network===

#===data for image difference version of 1 step prediction network===
def DataForOSAP_ID(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;        
    
    #===joints position===
    a_dir=rootdir+'/JointPos/video_'
    a_filenames=[a_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===joints position===
    
    #===network's input file names===    
    x_dir=rootdir+'/diff_frames/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/diff_frames/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)] 
    #===network's output file names===
    
    #===error mask file names===
    e_dir=rootdir+'/ErrorMasks_ID/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]  
    #===error mask file names===       
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
    a_train_filenames=list(np.asarray(a_filenames)[ind[0:n_train]])
    a_test_filenames=list(np.asarray(a_filenames)[ind[n_train:N]])       
    
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*im_h*im_w;y_size=(im_h,im_w)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    a_nb=4*42;a_size=(1,42)#each joint has 7 elements and we consider 6 of them
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)
    a_train_param=(a_train_filenames,a_nb,a_size)
    a_test_param=(a_test_filenames,a_nb,a_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch_OSAP(x_train_param,y_train_param,
                                                       e_train_param,
                                                       a_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch_OSAP(x_test_param,y_test_param,
                                                       e_test_param,
                                                       a_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for image difference version of 1 step prediction network===

#===data for Berkeley version of 1 step prediction network===
def DataForOSAP_BR(batch_size,p_train):
    rootdir=param.rootdir
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    im_h=param.im_h;im_w=param.im_w;        
    
    #===joints position===
    a_dir=rootdir+'/JointPos/video_'
    a_filenames=[a_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===joints position===
    
    #===network's input file names===    
    x_dir=rootdir+'/OpticalFlows/video_'
    x_filenames=[x_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(2,param.N_MOVE-6,3)]
    #===network's input file names===
    
    #===network's output file names===
    y_dir=rootdir+'/DonwSampledOF/video_'
    y_filenames=[y_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)] 
    #===network's output file names===
    
    #===error mask file names===
    #this will not be used
    e_dir=rootdir+'/ErrorMasks_ID/video_'
    e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
                 for vid in range(n_video) 
                 for frame in np.arange(5,param.N_MOVE-3,3)]  
    #===error mask file names===       
    
    #===split into train and test===
    N = len(x_filenames)
    n_train=int(p_train*N)
    np.random.seed(1234)
    ind=np.random.permutation(N)
    x_train_filenames=list(np.asarray(x_filenames)[ind[0:n_train]])
    x_test_filenames=list(np.asarray(x_filenames)[ind[n_train:N]])
    y_train_filenames=list(np.asarray(y_filenames)[ind[0:n_train]])
    y_test_filenames=list(np.asarray(y_filenames)[ind[n_train:N]])
    e_train_filenames=list(np.asarray(e_filenames)[ind[0:n_train]])
    e_test_filenames=list(np.asarray(e_filenames)[ind[n_train:N]])
    a_train_filenames=list(np.asarray(a_filenames)[ind[0:n_train]])
    a_test_filenames=list(np.asarray(a_filenames)[ind[n_train:N]])       
    
    x_nb=4*im_h*im_w;x_size=(im_h,im_w)
    y_nb=4*32*32;y_size=(32,32)    
    e_nb=4*im_h*im_w;e_size=(im_h,im_w)
    a_nb=4*42;a_size=(1,42)#each joint has 7 elements and we consider 6 of them
    
    x_train_param=(x_train_filenames,x_nb,x_size)
    x_test_param=(x_test_filenames,x_nb,x_size)
    y_train_param=(y_train_filenames,y_nb,y_size)
    y_test_param=(y_test_filenames,y_nb,y_size)
    e_train_param=(e_train_filenames,e_nb,e_size)
    e_test_param=(e_test_filenames,e_nb,e_size)
    a_train_param=(a_train_filenames,a_nb,a_size)
    a_test_param=(a_test_filenames,a_nb,a_size)    
    #===split into train and test===   

    images_train=createQueuedShuffledBatch_OSAP(x_train_param,y_train_param,
                                                       e_train_param,
                                                       a_train_param,
                                                       batch_size,shuffle=1)
    images_test=createQueuedShuffledBatch_OSAP(x_test_param,y_test_param,
                                                       e_test_param,
                                                       a_test_param,
                                                       param.TEST_BATCH_SIZE,
                                                       shuffle=0)    
    n_test=N-n_train
    return images_train,images_test,n_train,n_test       
#===data for Berkeley version of 1 step prediction network===

#===creat tf queue from filename lists for OSAP networks===
def createQueuedShuffledBatch_OSAP(x_param,y_param,e_param,a_param,
                              batch_size,shuffle):
    
    x_filenames,x_nb,x_shape=x_param        
    x_filename_queue=tf.train.string_input_producer(x_filenames,shuffle=False)    
    x_reader = tf.FixedLengthRecordReader(record_bytes=x_nb)   
    x_key, x_value = x_reader.read(x_filename_queue)
    x_record_bytes = tf.decode_raw(x_value, tf.float32)       
    x_im=tf.reshape(x_record_bytes,x_shape)
    
    y_filenames,y_nb,y_shape=y_param
    y_filename_queue=tf.train.string_input_producer(y_filenames,shuffle=False)    
    y_reader = tf.FixedLengthRecordReader(record_bytes=y_nb)   
    y_key, y_value = y_reader.read(y_filename_queue)
    y_record_bytes = tf.decode_raw(y_value, tf.float32)       
    y_im=tf.reshape(y_record_bytes,y_shape)
    
    e_filenames,e_nb,e_shape=e_param
    e_filename_queue=tf.train.string_input_producer(e_filenames,shuffle=False)    
    e_reader = tf.FixedLengthRecordReader(record_bytes=e_nb)   
    e_key, e_value = e_reader.read(e_filename_queue)
    e_record_bytes = tf.decode_raw(e_value, tf.float32)       
    e_im=tf.reshape(e_record_bytes,e_shape)
    
    a_filenames,a_nb,a_shape=a_param
    a_filename_queue=tf.train.string_input_producer(a_filenames,shuffle=False)    
    a_reader = tf.FixedLengthRecordReader(record_bytes=a_nb)   
    a_key, a_value = a_reader.read(a_filename_queue)
    a_record_bytes = tf.decode_raw(a_value, tf.float32)       
    a_im=tf.reshape(a_record_bytes,a_shape)
    
    if shuffle==1:    
        images = tf.train.shuffle_batch([x_im,y_im,e_im,a_im],
                                              batch_size,capacity=500,
                                              min_after_dequeue=100)
    else:
        images = tf.train.batch([x_im,y_im,e_im,a_im],
                                              batch_size,capacity=500)
    return images
#===creat tf queue from filename lists for OSAP networks===