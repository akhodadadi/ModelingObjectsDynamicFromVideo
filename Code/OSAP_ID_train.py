print('\014')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import pylab
from robot_input import DataForOSAP_ID
import parameters as param
from OneStepAheadPred_network import OSAP

tf.reset_default_graph()

#===some parameters===
batch_size=param.BATCH_SIZE
rootdir=param.rootdir
p_train=param.p_train
im_h=param.im_h;im_w=param.im_w;
#===some parameters===

#===load data and create tf queue===
images_train,images_test,n_train,n_test=DataForOSAP_ID(batch_size,
                                                   param.p_train)
n_batch_test=n_test/param.TEST_BATCH_SIZE
#===load data and create tf queue=== 

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
rbf_train=np.tile(rbf_mat0,[batch_size,1])
rbf_test=np.tile(rbf_mat0,[param.TEST_BATCH_SIZE,1])
fd_train={rbf_tf:rbf_train};fd_test={rbf_tf:rbf_test};
#===construct feed_dict===

#===define train_step,test and train error===
with tf.variable_scope('OSAP_ID') as scope:
    train_step,err_tarin_tf,y_train,_,_=OSAP(images_train,rbf_tf,'ID')
    scope.reuse_variables()
    _,err_test_tf,y_test,w_rbf,weights=OSAP(images_test,rbf_tf,'ID')
#===define train_step,test and train error===

#===initialize network===
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver()
#===initialize network===

#===training===
err_test=[];
saver.restore(sess, rootdir+'/trained_OSAP_ID.ckpt')
for epoch in range(5000): 
    
    if epoch%50==0:
        print('computing test error...')
        #---compute test error---
        e_test=0
        for i in range(n_batch_test):
            e_test+=sess.run(err_test_tf,feed_dict=fd_test)
            
        err_test.append(e_test/n_batch_test)
        #---compute test error---
        
        #---print status---
        t=datetime.datetime.now()
        print (t.ctime()+
        '...epoch %d,...error= %g' % (epoch,err_test[-1]))  
        #---print status---
        
    sess.run(train_step,feed_dict=fd_train)
#save_path = saver.save(sess, rootdir+'/trained_OSAP_ID.ckpt')
#===training===

im,yy,W=sess.run([images_test,y_test,weights],feed_dict=fd_test)
#W1,b1,W2,b2,Wr,br=W
y_pred=yy.reshape([100,64,64])
y_GT=im[1]
i=20;
plt.figure();pylab.imshow(y_GT[i,:,:]);
plt.figure();pylab.imshow(y_pred[i,:,:])

#===compute R2===
RSS=err_test[-1]
y_GT=[];e_m=[]
for i in range(n_batch_test):
    im=sess.run(images_test)    
    y_GT.append(im[1])
    e_m.append(im[2])
y_GT=np.concatenate(y_GT,0).reshape([n_batch_test*100,64,64])
e_m=np.concatenate(e_m,0).reshape([n_batch_test*100,64,64])

m_y_GT=np.tile(np.mean(y_GT,0),[y_GT.shape[0],1,1])
Syy=np.sum(np.square(y_GT-m_y_GT)*e_m)/(n_batch_test*100*64*64)
R2=1-(RSS/Syy)
#===compute R2===
