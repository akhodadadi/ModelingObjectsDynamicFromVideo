print('\014')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import pylab
from robot_input import DataForPCAN_OF
import parameters as param
from StatePredictionNetwork import SPN

tf.reset_default_graph()

#===some parameters===
batch_size=param.BATCH_SIZE
rootdir=param.rootdir
p_train=param.p_train
im_h=param.im_h;im_w=param.im_w;
#===some parameters===

#===load data and create tf queue===
images_train,images_test,n_train,n_test=DataForPCAN_OF(batch_size,
                                                   param.p_train)
n_batch_test=n_test/param.TEST_BATCH_SIZE
#===load data and create tf queue=== 

#===define train_step,test and train error===
n_input=3*param.N_command+param.N_PCA
n_state=param.N_PCA
with tf.variable_scope('PCAN_OF') as scope:
    train_step,err_tarin_tf,y_train,_=SPN(images_train,n_input,n_state)
    scope.reuse_variables()
    _,err_test_tf,y_test,weights=SPN(images_test,n_input,n_state)
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
for epoch in range(5000): 
    
    if epoch%50==0:
        print('computing test error...')
        #---compute test error---
        e_test=0
        for i in range(n_batch_test):
            e_test+=sess.run(err_test_tf)
            
        err_test.append(e_test/n_batch_test)
        #---compute test error---
        
        #---print status---
        t=datetime.datetime.now()
        print (t.ctime()+
        '...epoch %d,...error= %g' % (epoch,err_test[-1]))  
        #---print status--- 
        
    sess.run(train_step)
#save_path = saver.save(sess, rootdir+'/trained_DRBF.ckpt')
#===training===

#===compute R2===
RSS=err_test[-1]
y_GT=[]
for i in range(n_batch_test):
    y_GT.append(sess.run(images_test)[1])
y_GT=np.concatenate(y_GT,0).reshape([1800,param.N_PCA])
m_y_GT=np.tile(np.mean(y_GT,0),[y_GT.shape[0],1])
Syy=np.mean(np.sum(np.square(y_GT-m_y_GT),1))
R2=1-(RSS/Syy)
#===compute R2===


