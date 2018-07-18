from sklearn.decomposition import PCA
import numpy as np
import os
import parameters as param

rootdir=param.rootdir
n_components = 25
method='OF'

#===build the input matrix===
if method=='ID':#image difference
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    pix_list=[]
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
           d=rootdir+'/diff_frames/video_'+zz+'_frame_'+str(frame)
           pix_list.append(np.fromfile(d,'float32'))
           
    I=np.reshape(np.concatenate(pix_list),[len(pix_list),param.im_h*param.im_w])
    
else:#optical flow    
    n_video=len(os.listdir(os.path.join(rootdir,'pre_frames')))
    pix_list=[]
    for vid in range(n_video):  
        print('processing video=%d of %d'%(vid,n_video))
        zz='0'*(5-len(str(vid)))+str(vid)
        for frame in np.arange(2,param.N_MOVE-3,3):
           d=rootdir+'/OpticalFlows/video_'+zz+'_frame_'+str(frame)
           pix_list.append(np.fromfile(d,'float32'))
           
    I=np.reshape(np.concatenate(pix_list),[len(pix_list),param.im_h*param.im_w])
#===build the input matrix===
    
#===extract PCA and transform to PCA space=== 
print('computing PCA...')   
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(I)
#===extract PCA and transform to PCA space===

#===read joint position from file and save into a matrix===
print('read joint position from file...')
cp_mat=np.zeros([X_pca.shape[0],3*param.N_command])
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
#cp_mat=np.zeros([X_pca.shape[0],3*param.N_command])
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

#===concatenate and save as PCA Network's input and output===
print('save as State Prediction Networks input...')
pcan_in=np.concatenate([X_pca,cp_mat],1)
m_pcan_in=np.tile(np.max(np.abs(pcan_in),axis=0),[pcan_in.shape[0],1])
pcan_in=pcan_in/m_pcan_in

k=0
for vid in range(n_video):
    zz='0'*(5-len(str(vid)))+str(vid)
    for frame in np.arange(2,param.N_MOVE-3,3):
        if k%1000==0:
            print('video:%d,frame=%d'%(vid,frame))
        #input            
        d=rootdir+'/PCAN_'+method+'_input/video_'+zz+'_frame_'+str(frame)
        ss=pcan_in[k,:].reshape([1,pcan_in.shape[1]])
        np.asarray(ss,'float32').tofile(d)
        
        #output            
        d=rootdir+'/PCAN_'+method+'_output/video_'+zz+'_frame_'+str(frame)
        ss=X_pca[k,:].reshape([1,X_pca.shape[1]])
        np.asarray(ss,'float32').tofile(d)
        
        k+=1
#===concatenate and save as PCA Network's input and output===


