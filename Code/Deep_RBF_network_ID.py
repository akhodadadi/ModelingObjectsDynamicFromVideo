import tensorflow as tf
import parameters as param

def DRBF_ID(images,rbf_tf):
    #some parameters
    im_h=param.im_h;im_w=param.im_w;
    n_rbf=param.N_RBF
    std=.1;b=1
    
    #data
    x=images[0];
    n_r=tf.shape(x)[0]
    x=tf.reshape(x,shape=[n_r,im_h,im_w,1]);    
    y_=tf.reshape(images[0],shape=[n_r*im_h**2,1]);
    err_mask=tf.reshape(images[2],shape=[n_r*im_h**2,1]);
    

    #===construct network===
    #---conv layer 1---
    with tf.variable_scope('conv1') as scope:
        
        W1=tf.get_variable(name='W1',shape=[5,5,1,3],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b1 = tf.get_variable(name='b1',shape=[1],initializer=
                               tf.constant_initializer(value=b))
        conv1 = tf.nn.relu(tf.nn.conv2d(x,W1,[1,1,1,1],padding='SAME')+b1)
    #---conv layer 1---
    
    #---conv layer 2---
    with tf.variable_scope('conv2') as scope:
        
        W2=tf.get_variable(name='W2',shape=[5,5,3,3],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b2 = tf.get_variable(name='b2',shape=[1],initializer=
                               tf.constant_initializer(value=b))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,W2,[1,1,1,1],padding='SAME')+b2)
    #---conv layer 2---
    
    #---layer for computing RBF weights---
    flat_x=tf.contrib.layers.flatten(conv2)
    with tf.variable_scope('rbf') as scope:
        Wr=tf.get_variable(name='Wr',shape=[3*im_h**2,n_rbf],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        br=tf.get_variable(name='br',shape=[n_rbf],
                               initializer=tf.constant_initializer(value=b))

    w_rbf=tf.matmul(flat_x,Wr)+br
    #---layer for computing RBF weights---
    #===construct network===
    
    #===construct predicted image and compute error===
    #---tiling and reshaping w_rbf---
    w_rbf_tile=tf.reshape(tf.tile(w_rbf,[1,im_h**2]),
                          shape=[n_r*im_h**2,n_rbf])
    #---tiling and reshaping w_rbf---
    
    #---compute output of RBF---     
    rbf_out=tf.reshape(tf.reduce_sum(tf.mul(w_rbf_tile,rbf_tf),
                                     reduction_indices=1),shape=[n_r*im_h**2,1])
    #---compute output of RBF--- 
    
    #---reconstruct frame---
    y=rbf_out
    #---reconstruct frame---
    
    err=tf.reduce_mean(tf.reduce_sum(tf.mul(tf.square(y-y_),err_mask),
                                     reduction_indices=1))
    #===construct predicted image and compute error===
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(err)
    weights=(W1,b1,W2,b2,Wr,br)
    return train_step,err,y,w_rbf,weights
    