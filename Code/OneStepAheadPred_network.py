import tensorflow as tf
import parameters as param

def OSAP(images,rbf_tf,method):
    #some parameters
    n_fc1=100;n_fc2=100
    n_cc=n_fc1+42;#dimension after concatenating last fc layer and joint pos
    im_h=param.im_h;im_w=param.im_w;
    n_rbf=param.N_RBF
    std=.1;b=1
    
    #data
    x=images[0];
    n_r=tf.shape(x)[0]
    x=tf.reshape(x,shape=[n_r,im_h,im_w,1]);    
    y_=tf.reshape(images[1],shape=[n_r*im_h**2,1]);
    err_mask=tf.reshape(images[2],shape=[n_r*im_h**2,1]);
    jointPos=tf.reshape(images[3],shape=[n_r,42]);
    
    #===construct network===
    #---conv layer 1---
    with tf.variable_scope('conv1') as scope:
        
        W1=tf.get_variable(name='W1',shape=[5,5,1,10],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b1 = tf.get_variable(name='b1',shape=[1],initializer=
                               tf.constant_initializer(value=b))
        conv1 = tf.nn.relu(tf.nn.conv2d(x,W1,[1,1,1,1],padding='SAME')+b1)
    #---conv layer 1---
    
    #---conv layer 2---
    with tf.variable_scope('conv2') as scope:
        
        W2=tf.get_variable(name='W2',shape=[5,5,10,10],initializer=
                           tf.truncated_normal_initializer(stddev=std))        
        b2 = tf.get_variable(name='b2',shape=[1],initializer=
                               tf.constant_initializer(value=b))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,W2,[1,1,1,1],padding='SAME')+b2)
    #---conv layer 2---
    
    #---fc layer 1---
    flat_x=tf.contrib.layers.flatten(conv2)
    with tf.variable_scope('fc1') as scope:
        Wf1=tf.get_variable(name='Wf1',shape=[10*im_h**2,n_fc1],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        bf1=tf.get_variable(name='bf1',shape=[n_fc1],
                               initializer=tf.constant_initializer(value=b))

    h_fc1=tf.nn.relu(tf.matmul(flat_x,Wf1)+bf1)
    #---fc layer 1---
    
    #---fc layer 2---    
    with tf.variable_scope('fc2') as scope:
        Wf2=tf.get_variable(name='Wf2',shape=[n_fc1,n_fc2],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        bf2=tf.get_variable(name='bf2',shape=[n_fc2],
                               initializer=tf.constant_initializer(value=b))

    h_fc2=tf.nn.relu(tf.matmul(h_fc1,Wf2)+bf2)
    #---fc layer 2---
    
    #---concatenate with joint positions and batch normalize---
    h_conccat = tf.concat(1,[h_fc2,40*jointPos])
    mean,var=tf.nn.moments(h_conccat,[0])
    h_bn=tf.nn.batch_normalization(h_conccat,mean,var,0,1,1e-6)
    #---concatenate with joint positions---
    
    #---layer for computing RBF weights---
    with tf.variable_scope('rbf') as scope:
        Wr=tf.get_variable(name='Wr',shape=[n_cc,n_rbf],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        br=tf.get_variable(name='br',shape=[n_rbf],
                               initializer=tf.constant_initializer(value=b))

    if method=='OF':
        w_rbf=tf.nn.relu(tf.matmul(h_bn,Wr)+br)
    else:
        w_rbf=tf.matmul(h_bn,Wr)+br    
    #---layer for computing RBF weights---
    #===construct network===
    
    #===construct predicted image and compute error===
    #---tiling and reshaping w_rbf---
    w_rbf_tile=tf.reshape(tf.tile(w_rbf,[1,im_h**2]),
                          shape=[n_r*im_h**2,n_rbf])
    #---tiling and reshaping w_rbf---
    
    #---compute output of RBF---     
    y=tf.reshape(tf.reduce_sum(tf.mul(w_rbf_tile,rbf_tf),reduction_indices=1)
    ,shape=[n_r*im_h**2,1])
    #---compute output of RBF--- 
    
    err=tf.reduce_mean(tf.mul(tf.square(y-y_),err_mask))
    #===construct predicted image and compute error===
    
#    #===compute error for Y(t+1)=Y(t)===
#    xx=tf.reshape(x,shape=[n_r*im_h**2,1])
#    err_const=tf.reduce_mean(tf.reduce_sum(tf.mul(tf.square(xx-y_),err_mask),
#                                     reduction_indices=1))   
#    #===compute error for Y(t+1)=Y(t)===
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(err)
    weights=(W1,b1,W2,b2,Wr,br)
    return train_step,err,y,w_rbf,weights
    