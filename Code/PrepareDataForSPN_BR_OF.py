#this script computes and saves the output of a trained Berkeley net for all 
#frames in the training set. This output, then, is used as the input
#for training the action prediction network.

print('\014')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import pylab
from robot_input import createQueuedShuffledBatch
import parameters as param
from Berkeley_network import BRN

tf.reset_default_graph()

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

N = len(x_filenames)
x_nb=4*im_h*im_w;x_size=(im_h,im_w)
y_nb=4*32*32;y_size=(32,32)    
e_nb=4*im_h*im_w;e_size=(im_h,im_w)

x_param=(x_filenames,x_nb,x_size)
y_param=(y_filenames,y_nb,y_size)
e_param=(e_filenames,e_nb,e_size)

images = createQueuedShuffledBatch(x_param,y_param,e_param,1,0)

#===define train_step,test and train error===
with tf.variable_scope('BRN_OF') as scope:
    _,_,_,fc=BRN(images)
#===define train_step,test and train error===

#===initialize network===
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver()
#===initialize network===

#===compute output of trained network===
saver.restore(sess, rootdir+'/trained_BRN_OF.ckpt')
print('computing BRN output...')
fc_mat=np.zeros([N,32])
for i in range(N):
    if i%1000==0:
        print('i=%d of N=%d'%(i,N))
    fc_mat[i,:]=sess.run(fc)
#===compute output of trained network===

#===read joint position from file and save into a matrix===
print('read joint position from file...')
cp_mat=np.zeros([N,3*param.N_command])
k=0
for vid in range(n_video):
    zz='0'*(5-len(str(vid)))+str(vid)
    for frame in np.arange(2,param.N_MOVE-3,3):
        if k%1000==0:
            print('video:%d,frame=%d'%(vid,frame))
        d=rootdir+'/JointPos/video_'+zz+'_frame_'+str(frame)
        cp_mat[k,:]=np.fromfile(d,'float32').reshape([1,3*param.N_command])
        k+=1
#===read joint position from file and save into a matrix===

##===read commanded pose from file and save into a matrix===
#print('read commanded pose from file...')
#cp_mat=np.zeros([N,3*param.N_command])
#k=0
#for vid in range(n_video):
#    zz='0'*(5-len(str(vid)))+str(vid)
#    for frame in np.arange(2,param.N_MOVE-3,3):
#        if k%1000==0:
#            print('video:%d,frame=%d'%(vid,frame))
#        d=rootdir+'/CommandedPose/video_'+zz+'_frame_'+str(frame)
#        cp_mat[k,:]=np.fromfile(d,'float32').reshape([1,3*param.N_command])
#        k+=1
##===read commanded pose from file and save into a matrix===

##===read finger2d from file and save into a matrix===
#print('read finger2d from file...')
#cp_mat=np.zeros([N,3*param.N_command])
#k=0
#for vid in range(n_video):
#    zz='0'*(5-len(str(vid)))+str(vid)
#    for frame in np.arange(2,param.N_MOVE-3,3):
#        if k%1000==0:
#            print('video:%d,frame=%d'%(vid,frame))
#        d=rootdir+'/Finger2d/video_'+zz+'_frame_'+str(frame)
#        cp_mat[k,:]=np.fromfile(d,'float32').reshape([1,3*param.N_command])
#        k+=1
##===read finger2d from file and save into a matrix===

#===concatenate and save as State Prediction Network's input and output===
print('save as State Prediction Networks input and output...')
spn_in=np.concatenate([fc_mat,cp_mat],1)

m_spn_in=np.max(np.abs(spn_in),axis=0)
ind=np.where(m_spn_in==0);m_spn_in[ind]=1
m_spn_in=np.tile(m_spn_in,[spn_in.shape[0],1])
spn_in=spn_in/m_spn_in

k=0
for vid in range(n_video):
    zz='0'*(5-len(str(vid)))+str(vid)
    for frame in np.arange(2,param.N_MOVE-3,3):
        if k%1000==0:
            print('video:%d,frame=%d'%(vid,frame))
        
        #input
        d=rootdir+'/SPN_BR_OF_input/video_'+zz+'_frame_'+str(frame)
        ss=spn_in[k,:].reshape([1,spn_in.shape[1]])
        np.asarray(ss,'float32').tofile(d)
        
        #output
        d=rootdir+'/SPN_BR_OF_output/video_'+zz+'_frame_'+str(frame)
        ss=fc_mat[k,:].reshape([1,32])
        np.asarray(ss,'float32').tofile(d)
        
        k+=1
#===concatenate and save as State Prediction Network's input and output===

