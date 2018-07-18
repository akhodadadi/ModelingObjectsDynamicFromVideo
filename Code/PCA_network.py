import tensorflow as tf
import parameters as param

def PCAN(images):
    #some parameters
    nh1=500;nh2=500;nh3=param.N_PCA
    std=.1;b=1
    n_c=3*param.N_command+param.N_PCA
    
    #data
    x=images[0];
    n_r=tf.shape(x)[0]
    x=tf.reshape(x,shape=[n_r,n_c]);    
    y_=tf.reshape(images[1],shape=[n_r,param.N_PCA]);

    #===construct network===
    #---fc layer 1---
    with tf.variable_scope('fc1') as scope:
        W1=tf.get_variable(name='W1',shape=[n_c,nh1],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        b1=tf.get_variable(name='b1',shape=[nh1],
                               initializer=tf.constant_initializer(value=b))

    h1=tf.nn.relu(tf.matmul(x,W1)+b1)
    #---fc layer 1---
    
    #---fc layer 2---
    with tf.variable_scope('fc2') as scope:
        W2=tf.get_variable(name='W2',shape=[nh1,nh2],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        b2=tf.get_variable(name='b2',shape=[nh2],
                               initializer=tf.constant_initializer(value=b))

    h2=tf.nn.relu(tf.matmul(h1,W2)+b2)
    #---fc layer 2---
    
    #---fc layer 3---
    with tf.variable_scope('fc3') as scope:
        W3=tf.get_variable(name='W3',shape=[nh2,nh3],initializer=
                             tf.truncated_normal_initializer(stddev=std))
        b3=tf.get_variable(name='b3',shape=[nh3],
                               initializer=tf.constant_initializer(value=b))

    y=tf.matmul(h2,W3)+b3
    #---fc layer 2---
    #===construct network===
    
    #===compute error===    
    err=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_),reduction_indices=1))
    #===compute error===
    
    train_step = tf.train.AdamOptimizer(1e-2).minimize(err)
    weights=(W1,b1,W2,b2,W3,b3)
    return train_step,err,y,weights

    