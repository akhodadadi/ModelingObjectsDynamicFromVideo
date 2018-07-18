import numpy as np
import os
import cv2
import parameters as param
import tensorflow as tf

rootdir=param.rootdir
rawData_dir=os.path.join(rootdir,'TFrecords')
filenames=[rootdir+'/TFrecords/push_train.tfrecord-'+\
'0'*(5-len(str(i)))+str(i)+'-of-00264' for i in range(11)]

n_file=len(filenames)

def ComputeAndSaveOpticalFlow():
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))/param.N_MOVE
    for vid in range(n_video):        
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
            print('video:%d,frame=%d'%(vid,frame))
            d1=rootdir+'/frames/video_'+zz+'_frame_'+str(frame)
            I1=np.fromfile(d1,'float32').reshape([param.im_h,param.im_w])
            d2=rootdir+'/frames/video_'+zz+'_frame_'+str(frame+3)
            I2=np.fromfile(d2,'float32').reshape([param.im_h,param.im_w])
            of=cv2.\
            calcOpticalFlowFarneback(I1,I2,None,0.5, 3, 15, 3, 5, 1.2, 0)
            of_mag=of[:,:,0]**2 + of[:,:,1]**2
            d=rootdir+'/OpticalFlows/video_'+zz+'_frame_'+str(frame)            
            np.asarray(of_mag,'float32').tofile(d)
        

    
def ExtractReshapeSaveFrames():
    #===form desired TFrecord features dict===
    keys=['move/'+str(i)+'/image/encoded' for i in range(param.N_MOVE)]
    vals=param.N_MOVE*[tf.FixedLenFeature([], tf.string)]
    features_dict={keys[i]:vals[i] for i in range(len(keys))}
    features = CreateQueueFromTFrecord(features_dict)    
    images=[tf.image.decode_jpeg(features[keys[i]],channels=3)\
        for i in range(len(keys))]
    #===form desired TFrecord features dict===
    
    #===initialize tf===
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    #===initialize tf===
    
    #===read,reshape,save each frame===
    for i in range(10000):    
        try:
            print('i=%d'%(i))
            frames_list=sess.run([images])[0]
            k=0
            zz='0'*(5-len(str(i)))+str(i)
            for frame in frames_list:
                I=np.mean(np.asarray(frame,
                                     dtype='float32'),2)#reduce to 1 channel
                I=I[:,88:600]#crop to 512x512
                I=I[0:512:8,0:512:8]#subsample to 64x64
                d=rootdir+'/frames/video_'+zz+'_frame_'+str(k)
                np.asarray(I,dtype='float32').tofile(d)
                k+=1
        except tf.python.framework.errors.OutOfRangeError:            
            break
        #===read,reshape,save each frame===

def ComputeErrorMask():
    n_video=len(os.listdir(os.path.join(rootdir,'frames')))
    for vid in range(n_video):        
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3): 
            print('video:%d,frame=%d'%(vid,frame))
            d=rootdir+'/OpticalFlows/video_'+zz+'_frame_'+str(frame)
            I=np.fromfile(d,'float32').reshape([64,64])
            err_mask=np.ones(I.shape,dtype='float32')
            ind_white=np.where(I<2.0);i1,i2=ind_white
            ind_rand=np.random.choice(np.arange(i1.size),
                                      int(.95*i1.size),replace=False)
            err_mask[(i1[ind_rand],i2[ind_rand])]=0.0
            d=rootdir+'/ErrorMasks/video_'+zz+'_frame_'+str(frame)
            err_mask.tofile(d)

def ComputeErrorMask_ID():#error mask fordifference between pre and each frame    
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
           d=rootdir+'/diff_frames/video_'+zz+'_frame_'+str(frame)
           I=np.fromfile(d,'float32').reshape([64,64])
           err_mask=np.ones(I.shape,dtype='float32')
           ind_white=np.where(np.abs(I)<15.0);i1,i2=ind_white
           ind_rand=np.random.choice(np.arange(i1.size),
                                      int(.9*i1.size),replace=False)
           err_mask[(i1[ind_rand],i2[ind_rand])]=0.0
           d=rootdir+'/ErrorMasks_ID/video_'+zz+'_frame_'+str(frame)
           err_mask.tofile(d)        

def ExtractAndSaveJointPos():
    #to estimate of at t+1 we need the action at t,t+1 and t+2
    #===form desired TFrecord features dict===
    keys=['move/'+str(i)+'/joint/positions'\
          for i in range(param.N_MOVE)]
    vals=param.N_MOVE*[tf.FixedLenFeature([7], tf.float32)]
    features_dict={keys[i]:vals[i] for i in range(len(keys))}
    features = CreateQueueFromTFrecord(features_dict) 
    cp_tf=[features[keys[i]] for i in range(len(keys))]
    #===form desired TFrecord features dict===

    #===initialize tf===
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    #===initialize tf===  

    #===read joint position in each frame from TFrecord===
    cp_list=[]
    for i in range(10000):    
        try:
            print('i=%d'%(i))
            cp_list.append(sess.run(cp_tf))
        except tf.python.framework.errors.OutOfRangeError:            
            break
    cp_mat=np.array(cp_list)#shape=[n_video,n_move,7]    
    #===read joint position in each frame from TFrecord===
    
    #===save [a_(t:t+5)] for each frame===
    for vid in range(cp_mat.shape[0]):
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
            print('video:%d,frame=%d'%(vid,frame))
            cp=cp_mat[vid,frame:frame+6,:].reshape(1,42)
            d=rootdir+'/JointPos/video_'+zz+'_frame_'+str(frame)
            np.asarray(cp,dtype='float32').tofile(d)
    #===save [a_t,a_(t+1),a_(t+2)] for each frame===            
            
def ExtractAndSaveCommandedPose():
    #to estimate of at t+1 we need the action at t,t+1 and t+2
    #===form desired TFrecord features dict===
    keys=['move/'+str(i)+'/commanded_pose/vec_pitch_yaw'\
          for i in range(param.N_MOVE)]
    vals=param.N_MOVE*[tf.FixedLenFeature([5], tf.float32)]
    features_dict={keys[i]:vals[i] for i in range(len(keys))}
    features = CreateQueueFromTFrecord(features_dict) 
    cp_tf=[features[keys[i]] for i in range(len(keys))]
    #===form desired TFrecord features dict===

    #===initialize tf===
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    #===initialize tf===  

    #===read commanded pose in each frame from TFrecord===
    cp_list=[]
    for i in range(10000):    
        try:
            print('i=%d'%(i))
            cp_list.append(sess.run(cp_tf))
        except tf.python.framework.errors.OutOfRangeError:            
            break
    cp_mat=np.array(cp_list)#shape=[n_video,n_move,5]    
    #===read commanded pose in each frame from TFrecord===
    
    #===save [a_t,a_(t+1),a_(t+2)] for each frame===
    for vid in range(cp_mat.shape[0]):
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
            print('video:%d,frame=%d'%(vid,frame))
            cp=cp_mat[vid,[frame,frame+1,frame+4],:].reshape(1,15)
            d=rootdir+'/CommandedPose/video_'+zz+'_frame_'+str(frame)
            np.asarray(cp,dtype='float32').tofile(d)
    #===save [a_t,a_(t+1),a_(t+2)] for each frame===
    
def ExtractAndSaveFinger2d():
    #to estimate of at t+1 we need the action at t,t+1 and t+2
    #===form desired TFrecord features dict===
    keys=['finger2d_'+str(i) for i in range(param.N_MOVE)]
    vals=param.N_MOVE*[tf.FixedLenFeature([2], tf.float32)]
    features_dict={keys[i]:vals[i] for i in range(len(keys))}
    features = CreateQueueFromTFrecord(features_dict) 
    cp_tf=[features[keys[i]] for i in range(len(keys))]
    #===form desired TFrecord features dict===

    #===initialize tf===
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    #===initialize tf===  

    #===read finger2d in each frame from TFrecord===
    cp_list=[]
    for i in range(10000):    
        try:
            print('i=%d'%(i))
            cp_list.append(sess.run(cp_tf))
        except tf.python.framework.errors.OutOfRangeError:            
            break
    cp_mat=np.array(cp_list)#shape=[n_video,n_move,5]    
    #===read finger2d in each frame from TFrecord===
    
    #===save [a_t,a_(t+1),a_(t+2)] for each frame===
    for vid in range(cp_mat.shape[0]):
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
            print('video:%d,frame=%d'%(vid,frame))
            cp=cp_mat[vid,[frame,frame+1,frame+4],:].reshape(1,6)
            d=rootdir+'/Finger2d/video_'+zz+'_frame_'+str(frame)
            np.asarray(cp,dtype='float32').tofile(d)
    #===save [a_t,a_(t+1),a_(t+2)] for each frame===              

              
def ExtractAndSavePreFrame():
    #===form desired TFrecord features dict===
    keys=['pre/image/encoded']
    vals=[tf.FixedLenFeature([], tf.string)]
    features_dict={keys[i]:vals[i] for i in range(len(keys))}
    features = CreateQueueFromTFrecord(features_dict)    
    images=[tf.image.decode_jpeg(features[keys[i]],channels=3)\
        for i in range(len(keys))]
    #===form desired TFrecord features dict===
    
    #===initialize tf===
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    #===initialize tf===
    
    #===read,reshape,save each frame===
    for i in range(10000):    
        try:
            print('i=%d'%(i))
            frame=sess.run(images)[0]
            zz='0'*(5-len(str(i)))+str(i)
            I=np.mean(np.asarray(frame,
                                 dtype='float32'),2)#reduce to 1 channel
            I=I[:,88:600]#crop to 512x512
            I=I[0:512:8,0:512:8]#subsample to 64x64
            d=rootdir+'/pre_frames/video_'+zz
            np.asarray(I,dtype='float32').tofile(d)
        except tf.python.framework.errors.OutOfRangeError:            
            break
    #===read,reshape,save each frame===    

def ComputeDiffBetweenPreAndFrame():
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        d=rootdir+'/pre_frames/video_'+zz
        pre_frame=np.fromfile(d,'float32')
        for frame in range(param.N_MOVE):
           d=rootdir+'/frames/video_'+zz+'_frame_'+str(frame)
           I=np.fromfile(d,'float32')
           I_diff=np.reshape(pre_frame-I,[64,64])
           d=rootdir+'/diff_frames/video_'+zz+'_frame_'+str(frame)
           np.asarray(I_diff,dtype='float32').tofile(d)

def GenrateDownSampleFrames():
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in range(param.N_MOVE):
           d=rootdir+'/frames/video_'+zz+'_frame_'+str(frame)
           I=np.fromfile(d,'float32').reshape([64,64])
           I_ds=I[0:64:2,0:64:2]
           d=rootdir+'/DonwSampledframes/video_'+zz+'_frame_'+str(frame)
           np.asarray(I_ds,dtype='float32').tofile(d)
 
def GenrateDownSampleOF():
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
           d=rootdir+'/OpticalFlows/video_'+zz+'_frame_'+str(frame)
           I=np.fromfile(d,'float32').reshape([64,64])
           I_ds=I[0:64:2,0:64:2]
           d=rootdir+'/DonwSampledOF/video_'+zz+'_frame_'+str(frame)
           np.asarray(I_ds,dtype='float32').tofile(d)
             
def CreateQueueFromTFrecord(features_dict):
    filename_queue=tf.train.string_input_producer(filenames,shuffle=False,
                                              num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features=features_dict)
    return features   


              
#
#        
#n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
#p_list=[]
#for vid in range(n_video):  
#    print('processing video=%d of %d'%(vid,n_video))
#    zz='0'*(5-len(str(vid)))+str(vid)
#    for frame in np.arange(2,param.N_MOVE-3,3):
#       d=rootdir+'/diff_frames/video_'+zz+'_frame_'+str(frame)
#       p_list.append(np.fromfile(d,dtype='float32'))

        