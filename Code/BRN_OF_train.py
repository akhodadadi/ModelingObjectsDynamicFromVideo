print('\014')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import pylab
from robot_input import DataForBRN_OF
import parameters as param
from Berkeley_network import BRN

tf.reset_default_graph()

#===some parameters===
batch_size=param.BATCH_SIZE
rootdir=param.rootdir
p_train=param.p_train
im_h=param.im_h;im_w=param.im_w;
#===some parameters===

#===load data and create tf queue===
images_train,images_test,n_train,n_test=DataForBRN_OF(batch_size,
                                                   param.p_train)
n_batch_test=n_test/param.TEST_BATCH_SIZE
#===load data and create tf queue=== 

#===define train_step,test and train error===
with tf.variable_scope('DRBF') as scope:
    train_step,err_tarin_tf,y_train,_,_=BRN(images_train)
    scope.reuse_variables()
    _,err_test_tf,y_test,fc,conv2=BRN(images_test)
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
saver.restore(sess, rootdir+'/trained_BRN_OF.ckpt')

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
save_path = saver.save(sess, rootdir+'/trained_BRN_OF.ckpt')
#===training===

im,y_pred,FC,conv=sess.run([images_test,y_test,fc,conv2])
y_pred=y_pred.reshape([100,32,32])
y_GT=im[1]

i=5;
plt.close('all')
plt.figure();pylab.imshow(y_pred[i,:,:])
plt.figure();pylab.imshow(y_GT[i,:,:]);


plt.figure();
for j in range(16):
    plt.subplot(4,4,j+1)
    pylab.imshow(conv[i,:,:,j])
    plt.xticks([],[]);plt.yticks([],[]);
    
#===compute R2===
RSS=np.mean(err_test[-100:])
y_GT=[];e_m=[]
for i in range(n_batch_test):
    im=sess.run(images_test)    
    y_GT.append(im[1])
y_GT=np.concatenate(y_GT,0).reshape([n_batch_test*100,32,32])

m_y_GT=np.tile(np.mean(y_GT,0),[y_GT.shape[0],1,1])
Syy=np.sum(np.square(y_GT-m_y_GT))/(n_batch_test*100*32*32)
R2=1-(RSS/Syy)
#===compute R2===

