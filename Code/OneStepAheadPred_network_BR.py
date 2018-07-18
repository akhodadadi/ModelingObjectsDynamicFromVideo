import tensorflow as tf
import parameters as param
import numpy as np

def OSAP_BR(images):
    #some parameters
    nc1=10;nc2=16#no. of channels
    n_cc=74;#dimension after concatenating last fc layer and joint pos
    im_h=param.im_h;im_w=param.im_w;
    std=.0001;b=.001
    
    #data
    x=images[0];
    n_r=tf.shape(x)[0]
    x=tf.reshape(x,shape=[n_r,im_h,im_w,1]);    
    y_=tf.reshape(images[1],shape=[n_r*32*32,1]);
    jointPos=tf.reshape(images[3],shape=[n_r,42]);
    
    #===generate ind_x and ind_y for computing fc===
    x_grid,y_grid=np.meshgrid(range(64),range(64))
    ind_x=np.reshape(x_grid,[64**2,1])
    ind_y=np.reshape(y_grid,[64**2,1])
    ind_x_tf=tf.constant(ind_x,dtype=tf.float32)
    ind_y_tf=tf.constant(ind_y,dtype=tf.float32)
    #===generate ind_x and ind_y for computing fc===

    #===construct network===
    #---conv layer 1---
    with tf.variable_scope('conv1') as scope:
        
        W1=tf.get_variable(name='W1',shape=[5,5,1,nc1],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b1 = tf.get_variable(name='b1',shape=[nc1],initializer=
                               tf.constant_initializer(value=b))
        conv1 = tf.nn.relu(tf.nn.conv2d(x,W1,[1,1,1,1],padding='SAME')+b1)
    #---conv layer 1---
    
    #---conv layer 2---
    with tf.variable_scope('conv2') as scope:
        
        W2=tf.get_variable(name='W2',shape=[5,5,nc1,nc2],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b2 = tf.get_variable(name='b2',shape=[nc2],initializer=
                               tf.constant_initializer(value=b))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,W2,[1,1,1,1],padding='SAME')+b2)
    #---conv layer 2---    
    
    #---spatial softmax---
    conv_trans=tf.transpose(tf.transpose(conv2,perm=[0,1,3,2]),
                               perm=[0,2,1,3])
    conv_reshaped=tf.reshape(conv_trans,shape=[n_r*nc2,im_h*im_w])
    spat_soft=tf.nn.softmax(conv_reshaped)#shape=[n_r*nc2,im_h*im_w]
    #---spatial softmax---
    
    #---compute fc (see page 4)---
    fc_x=tf.matmul(spat_soft,ind_x_tf)
    fc_y=tf.matmul(spat_soft,ind_y_tf)
    fc = tf.reshape(tf.concat(1,[fc_x,fc_y]),shape=[n_r,32])
    #---compute fc (see page 4)---

    #---concatenate with joint positions---
    h_conccat = tf.concat(1,[fc,255*jointPos])
    #---concatenate with joint positions---
    
    #---fully connected layer for reconstructinig down-sampled image---
    with tf.variable_scope('fc1') as scope:
        Wf1=tf.get_variable(name='Wf1',shape=[n_cc,32**2],initializer=
                             tf.truncated_normal_initializer(stddev=1e-5))
        bf1=tf.get_variable(name='bf1',shape=[32**2],
                               initializer=tf.constant_initializer(value=0))

    y=tf.reshape(tf.nn.relu(tf.matmul(h_conccat,Wf1)+bf1),shape=[n_r*32*32,1])
    #---fully connected layer for reconstructinig down-sampled image---    
    #===construct network===
    
    #===compute error===     
    err=tf.reduce_mean(tf.square(y-y_))
    #===error===
    
#    #===compute error for Y(t+1)=Y(t)===
#    xx=tf.reshape(x,shape=[n_r*im_h**2,1])
#    err_const=tf.reduce_mean(tf.reduce_sum(tf.mul(tf.square(xx-y_),err_mask),
#                                     reduction_indices=1))   
#    #===compute error for Y(t+1)=Y(t)===
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(err)
    weights=(W1,b1,W2,b2,Wf1,bf1)
    return train_step,err,y,h_conccat
    