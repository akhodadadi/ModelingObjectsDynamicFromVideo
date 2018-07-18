#this script computes and saves the output of a trained DRBF for all 
#frames in the training set. This output, then, is used as the input
#for training the action prediction network.

print('\014')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import pylab
import parameters as param
from Deep_RBF_network_OF import DRBF_OF
from robot_input import createQueuedShuffledBatch
import os
import copy

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
y_filenames=copy.copy(x_filenames)
#===network's output file names===

#===error masks file names===
e_dir=rootdir+'/ErrorMasks/video_'
e_filenames=[e_dir+'0'*(5-len(str(vid)))+str(vid)+'_frame_'+str(frame)\
             for vid in range(n_video) 
             for frame in np.arange(2,param.N_MOVE-3,3)]    
#===error masks file names===

N = len(x_filenames)
x_nb=4*im_h*im_w;x_size=(im_h,im_w)
y_nb=4*im_h*im_w;y_size=(im_h,im_w)    
e_nb=4*im_h*im_w;e_size=(im_h,im_w)

x_param=(x_filenames,x_nb,x_size)
y_param=(y_filenames,y_nb,y_size)
e_param=(e_filenames,e_nb,e_size)

images = createQueuedShuffledBatch(x_param,y_param,e_param,1,0)

#===compute RBF Gaussian kernels===
tau=1/(7.0**2)
mu_x,mu_y=np.meshgrid(np.linspace(0,64,np.sqrt(param.N_RBF)),
                      np.linspace(0,64,np.sqrt(param.N_RBF)))
x_grid,y_grid=np.meshgrid(np.arange(im_h),np.arange(im_w))
z_grid=np.hstack([x_grid.reshape((im_h*im_w),1),
             y_grid.reshape((im_h*im_w),1)])
rbf_mat0=np.zeros([im_h*im_w,param.N_RBF])
k=0
for i in range(mu_x.shape[0]):
    for j in range(mu_x.shape[1]):
        mu=np.tile(np.hstack([mu_x[i,j],mu_y[i,j]]),[z_grid.shape[0],1]);
        z_mu=z_grid-mu
        mu_S_mu=np.sum(tau*z_mu*z_mu,1)
        rbf_mat0[:,k]=np.exp(-mu_S_mu)
        k+=1
        
rbf_tf=tf.placeholder(dtype=tf.float32,shape=[None,param.N_RBF])        
#===compute RBF Gaussian kernels===

#===construct feed_dict===
fd={rbf_tf:rbf_mat0};
#===construct feed_dict===

#===define train_step,test and train error===
with tf.variable_scope('DRBF') as scope:
    _,_,Y,w_rbf,_=DRBF_OF(images,rbf_tf)
#===define train_step,test and train error===

#===initialize network===
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver()
#===initialize network===

#===compute output of trained network===
saver.restore(sess, rootdir+'/trained_DRBF_OF.ckpt')
print('computing DRBF output...')
w_rbf_mat=np.zeros([N,param.N_RBF])
for i in range(N):
    if i%1000==0:
        print('i=%d of N=%d'%(i,N))
    w_rbf_mat[i,:]=sess.run(w_rbf,feed_dict=fd)
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
spn_in=np.concatenate([w_rbf_mat,cp_mat],1)

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
        d=rootdir+'/SPN_OF_input/video_'+zz+'_frame_'+str(frame)
        ss=spn_in[k,:].reshape([1,spn_in.shape[1]])
        np.asarray(ss,'float32').tofile(d)
        
        #output
        d=rootdir+'/SPN_OF_output/video_'+zz+'_frame_'+str(frame)
        ss=w_rbf_mat[k,:].reshape([1,param.N_RBF])
        np.asarray(ss,'float32').tofile(d)
        
        k+=1
#===concatenate and save as State Prediction Network's input and output===

