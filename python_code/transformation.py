from __future__ import division
import os,cv2,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from spatial_transformer import transformer
import numpy as np

training_phase = False
if(training_phase):
    rgb_path = "../traindata/RGB512Full/"
    mat_path = "../traindata/transform/transform_512/"
    label_path = "../traindata/label_refine/"
    save_model_path = "../trainedmodels/transform"
    num_folder = 2
else:
    save_model_path = "../trainedmodels/transform"
    mat_path = "../testdata/transform/transform_512/"
    save_result_path = "../result/transform/transform_512/"
    label_path = "../testdata/label_refine_512/"

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

batch_size = 1
num_proposal = 10

crop_size_h = 512
crop_size_w = 1024
sp = 512

#resolution 256 x 512
'''
crop_size_h = 512
crop_size_w = 1024
sp = 512
'''
# resolution 1024 x 2048
'''
crop_size_h = 1024
crop_size_w = 2048
sp = 1024
'''


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)



def lrelu(x):
    return tf.maximum(0.2*x,x)


MEAN_VALUES = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))


def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def one_hot(label):

    output=np.zeros((1,100,label.shape[0],label.shape[1]),dtype=np.float32);
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]>0:
                output[0,label[i,j]-1,i,j]=1
    return output

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias


def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('Models/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-MEAN_VALUES
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    net['pool5']=build_net('pool',net['conv5_4'])
    return net
def proposal_transform(proposal,mask,label):

    input = tf.concat([proposal,mask,tf.tile(label,[num_proposal,1,1,1])],axis = 3)

    if(sp==512):
        input = tf.image.resize_bilinear(input,[crop_size_h//2, crop_size_w//2], align_corners = True)
    elif(sp==1024):
        input = tf.image.resize_bilinear(input,[crop_size_h//4, crop_size_w//4], align_corners = True)

    conv1_transform = slim.repeat(input, 2, slim.conv2d, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_transform_conv1')
    pool1_transform = slim.avg_pool2d(conv1_transform, [3, 3], stride=2, padding='SAME', scope='g_transform_pool1')

    conv2_transform = slim.repeat(pool1_transform, 2, slim.conv2d, 128, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_transfrom_conv2')
    pool2_transform = slim.max_pool2d(conv2_transform, [3, 3], stride=2, padding='SAME', scope='g_transform_pool2')

    conv3_transform = slim.repeat(pool2_transform, 3, slim.conv2d, 256, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_transform_conv3')
    pool3_transform = slim.max_pool2d(conv3_transform, [3, 3], stride=2, padding='SAME', scope='g_transform_pool3')

    conv4_transform = slim.repeat(pool3_transform, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_transform_conv4')
    pool4_transform = slim.max_pool2d(conv4_transform, [3, 3], stride=2, padding='SAME', scope='g_transform_pool4')

    conv5_transform = slim.repeat(pool4_transform, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_transform_conv5')
    pool5_transform = slim.max_pool2d(conv5_transform, [3, 3], stride=2, padding='SAME', scope='g_transform_pool5')

    pool6_transform = slim.avg_pool2d(pool5_transform, [8, 16], stride=[8, 16], padding='VALID',
                                      scope='g_transform_pool6')

    pool6_reshape = tf.reshape(pool6_transform,[batch_size*num_proposal,512])
    initial_W = np.zeros((512,6),dtype = np.float32)
    W_fc1 = tf.Variable(initial_value = initial_W, name='g_transform_param')

    initial = np.array([[1.0, 0., 0.], [0., 1.0, 0.]])
    initial = initial.astype('float32')
    initial = initial.flatten()

    b_fc1 = tf.Variable(initial_value=initial, name='g_b_fc1')
    transform_param = tf.matmul(pool6_reshape, W_fc1) + b_fc1
    # This is to avoid bilinear intepolation to corrupt the original pixel in cityscape dataset.
    # Since cityscape has very similar viewpoint patterns, spatial transformer can only slightly influence the result
    # The following two lines can be comment if on other datasets like NYU and ADE20k
    if(not training_phase):
        transform_param = tf.round(transform_param)

    proposal_refine = transformer(proposal,transform_param,[crop_size_h,crop_size_w])

    proposal_refine_new = tf.multiply(proposal_refine,tf.tile(mask,[1,1,1,3]))
    proposal_refine_new = tf.reduce_sum(proposal_refine_new,axis = 0, keep_dims = True)

    return proposal_refine,proposal_refine_new

def compute_error(real,fake,label,mask,flag = True):
    if (flag):
        return tf.reduce_sum(tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[0,3]),-1),reduction_indices=[0,1,2]))

    loss = tf.reduce_mean(tf.abs(fake-real),reduction_indices =[3],keep_dims = True)
    loss = tf.multiply(loss,mask)
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(mask)+1.0)

    return loss




with tf.variable_scope(tf.get_variable_scope()) as scope:
    label=tf.placeholder(tf.float32,[batch_size,None,None,20])
    real_image=tf.placeholder(tf.float32,[batch_size,None,None,3])
    fake_image=tf.placeholder(tf.float32,[batch_size,None,None,3])

    mask = tf.placeholder(tf.float32,[batch_size*num_proposal,crop_size_h,crop_size_w,1])
    proposal = tf.placeholder(tf.float32,[batch_size*num_proposal,crop_size_h,crop_size_w,3])


    proposal_transfer,proposal_new = proposal_transform(proposal,mask,label)

    real_B,real_G,real_R = tf.split(real_image,num_or_size_splits = 3,axis = 3)
    real_image_new = tf.concat([real_R, real_G,real_B],axis = 3)
    weight=tf.placeholder(tf.float32)



    mask_all = tf.reduce_sum(mask, axis=0, keep_dims = True)
    p_transformer = compute_error(real_image_new,proposal_new*255.0,label,mask_all, False)

    G_loss = p_transformer*1.0



lr=tf.placeholder(tf.float32)


t_vars = tf.trainable_variables()
for var in t_vars:
    if var.name.startswith('g_'):
        print var.name
if(training_phase):
    G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss = G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(save_model_path+ "/0100")


if ckpt:
   print('loaded '+ckpt.model_checkpoint_path)
   saver.restore(sess,ckpt.model_checkpoint_path)
if(training_phase):
    best=np.inf
    all_loss=np.zeros(3000,dtype=float)
    g_loss=np.zeros(3000,dtype=float)
    all_c_loss=np.zeros(3000,dtype=float)
    input_images=[None]*3000
    label_images=[None]*3000

    tmp_label = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
    tmp_image = np.zeros((batch_size, crop_size_h,crop_size_w, 3), dtype = float)
    tmp_mask = np.zeros((batch_size*num_proposal,crop_size_h, crop_size_w,1), dtype =float)
    #tmp_mask_all = np.zeros((batch_size,crop_size_h,crop_size_w,1),dtype = float)
    tmp_proposal = np.zeros((batch_size*num_proposal,crop_size_h,crop_size_w,3),dtype = float)
    total_number = 2975
    base_lr = 1e-4
    for epoch in range(1,101):
        if os.path.isdir(save_model_path+"/%04d"%epoch):
            continue
        cnt=0

        tmp_list = np.random.permutation(total_number) + 1
        global_ind = 0
        ind = 0

        while(global_ind<total_number):
            st=time.time()
            tmp_label = np.zeros((batch_size, 19, crop_size_h, crop_size_w), dtype=float)
            tmp_image = np.zeros((batch_size, crop_size_h, crop_size_w, 3), dtype=float)
            tmp_mask = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 1), dtype=float)

            tmp_proposal = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 3), dtype=float)
            for batch_count in range(0, batch_size):
                if(ind>=total_number):
                    ind = 0

                rand_folder = np.random.permutation(num_folder) + 1
                try:
                    dic = scipy.io.loadmat(mat_path +"%02d"%rand_folder[0]+ "/%08d.mat"%tmp_list[ind])
                except:
                    print("cannot load" + "%08d.mat" % tmp_list[ind])
                    print(mat_path +"%02d"%rand_folder[0]+ "/%08d.mat"%tmp_list[ind])
                    ind = ind + 1
                    global_ind = global_ind + 1
                    continue
                cnt += 1
                num = dic['proposal'].shape[0]

                tmp_label[:, :, :, :] = helper.get_semantic_map(label_path+ "/%08d.png" % tmp_list[ind])
                tmp_image[:, :, :, :] = np.expand_dims(np.float32(cv2.imread(rgb_path + "/%08d.png"%tmp_list[ind])),axis=0)
                if(num<=num_proposal):
                    tmp_mask[0:num, :, :, :] = np.expand_dims(dic['mask'], axis=3).astype(np.float32)
                    tmp_proposal[0:num, :, :, :] = (dic['proposal']).astype(np.float32)/ 255.0
                if(num>num_proposal):
                    d = np.random.permutation(num)
                    tmp_mask[:,:,:,:] = np.expand_dims(dic['mask'],axis=3).astype(np.float32)[d[0:num_proposal],:,:,:]#+np.expand_dims(np.expand_dims(dic2['mask'],axis=0),axis=3)
                    tmp_proposal[:,:,:,:] = (dic['proposal']).astype(np.float32)[d[0:num_proposal],:,:,:]/255.0 #+ np.expand_dims(dic2['proposal'],axis = 0)

                z = np.concatenate((tmp_label, np.expand_dims(1 - np.sum(tmp_label, axis=1), axis=1)), axis=1)
                z = z.transpose([0,2,3,1])
               # print z.shape
                ind = ind + 1
                global_ind = global_ind + 1
            _,G_current,l6,lr_value=sess.run([G_opt,G_loss,p_transformer,lr],feed_dict={label:z,real_image:tmp_image,mask: tmp_mask, proposal:tmp_proposal,lr:min(base_lr*np.power(1.1,epoch-1),1e-5 if epoch>112 else 1e-5)})
            g_loss[cnt]=G_current

            print("%d %d %.2f %.2f %.2f %.6f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l6),time.time()-st, lr_value))


        if epoch%1==0:
            os.makedirs(save_model_path+"/%04d" % epoch)
            target = open(save_model_path+"/%04d/score.txt" % epoch, 'w')
            target.write("%f" % np.mean(g_loss[np.where(g_loss)]))
            target.close()
            saver.save(sess,save_model_path+"/model.ckpt")
        if epoch%20==0:
            saver.save(sess,save_model_path+"/%04d/model.ckpt"%epoch)
        cnt=3000
        tmp_list = range(2926,2976) + range(100001,100051)
        #tmp_list = range(1100,1200)
        global_count = 0
        semantic = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
        proposal_val = np.zeros((batch_size*num_proposal,crop_size_h,crop_size_w,3),dtype = float)
        proposal_mask = np.zeros((batch_size*num_proposal,crop_size_h,crop_size_w,1),dtype = float)

        while (global_count < 99):
            semantic = np.zeros((batch_size, 19, crop_size_h, crop_size_w), dtype=float)
            proposal_val = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 3), dtype=float)
            proposal_mask = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 1), dtype=float)
            for local_count in range(0,batch_size):
                semantic[:,:,:,:]= helper.get_semantic_map(label_path + "/%08d.png" % tmp_list[global_count])

                dic = scipy.io.loadmat(mat_path + "/01/"+"%08d.mat"%tmp_list[global_count])
                num = dic['proposal'].shape[0]
                if (num <= num_proposal):
                    proposal_mask[0:num, :, :, :] = np.expand_dims(dic['mask'], axis=3).astype(np.float32)
                    proposal_val[0:num, :, :, :] = (dic['proposal']).astype(np.float32) / 255.0
                if (num > num_proposal):
                    d = np.random.permutation(num)
                    proposal_mask[:, :, :, :] = np.expand_dims(dic['mask'], axis=3).astype(np.float32)[d[0:num_proposal], :, :,:]  # +np.expand_dims(np.expand_dims(dic2['mask'],axis=0),axis=3)
                    proposal_val[:, :, :, :] = (dic['proposal']).astype(np.float32)[d[0:num_proposal], :, :,:] / 255.0  # + np.expand_dims(dic2['proposal'],axis = 0)

                global_count = global_count +1
            z = np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=1), axis=1)), axis=1)
            z = z.transpose([0, 2, 3, 1])

            proposal_output=sess.run(proposal_new,feed_dict={label:z,proposal:proposal_val,mask:proposal_mask})

            proposal_output = np.minimum(np.maximum(proposal_output, 0.0), 1.0)
            proposal_output = proposal_output * 255.0
            proposal_output_final = np.concatenate((np.expand_dims(proposal_output[:, :, :, 2], axis=3),
                                                    np.expand_dims(proposal_output[:, :, :, 1], axis=3),
                                                    np.expand_dims(proposal_output[:, :, :, 0], axis=3)), axis=3)

            cv2.imwrite(save_model_path + "/%04d/%06d_output_proposal.png" % (epoch, cnt), np.uint8(proposal_output_final[0, :, :, :]))

            print(cnt)
            cnt = cnt + batch_size
else:

    if not os.path.isdir(save_result_path):
        os.makedirs(save_result_path)



    semantic = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
    proposal_val = np.zeros((batch_size*num_proposal,crop_size_h,crop_size_w,3),dtype = float)
    proposal_mask = np.zeros((batch_size*num_proposal,crop_size_h,crop_size_w,1),dtype = float)
    for ind in range(100001,100501):
        print(ind)
        semantic = np.zeros((batch_size, 19, crop_size_h, crop_size_w), dtype=float)
        proposal_val = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 3), dtype=float)
        proposal_mask = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 1), dtype=float)
        semantic[:, :, :, :] = helper.get_semantic_map(label_path + "/%08d.png" % ind)

        dic = scipy.io.loadmat(mat_path + "/%08d.mat" % (ind))

        proposals = dic['proposal'].astype(np.float32) / 255.0

        query_mask = dic['mask'].astype(np.float32)

        if (proposals.ndim == 0):
            continue
        if (proposals.ndim == 3 and query_mask.ndim == 2):
            proposals = np.expand_dims(proposals, axis=0)
            query_mask = np.expand_dims(query_mask, axis=0)
        proposal_transfered_result = np.zeros((proposals.shape[0],proposals.shape[1],proposals.shape[2],proposals.shape[3]),dtype=np.float32)
        total_proposal = proposals.shape[0]
        print((np.floor(total_proposal / num_proposal) + 1).astype(np.int32))
        for n in range(0,(np.floor(total_proposal/num_proposal)+1).astype(np.int32)):
            proposal_val = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 3), dtype=float)
            proposal_mask = np.zeros((batch_size * num_proposal, crop_size_h, crop_size_w, 1), dtype=float)
            if(total_proposal<=num_proposal):
                proposal_val[0:total_proposal, :, :, :] = proposals[:,:,:,:]
                proposal_mask[0:total_proposal, :, :, :] = np.expand_dims(query_mask,axis = 3)
                proposal_transfer_data=sess.run(proposal_transfer,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=1),axis=1)),axis=1).transpose([0,2,3,1]),proposal:proposal_val,mask:proposal_mask})
                proposal_transfered_result[0:total_proposal,:,:,:] = proposal_transfer_data[0:total_proposal,:,:,:]
            elif(n==np.floor(total_proposal/num_proposal)):
                residual_num = total_proposal - num_proposal*n
                start_num = num_proposal*n
                proposal_val[0:residual_num,:,:,:] = proposals[start_num:total_proposal,:,:,:]
                proposal_mask[0:residual_num,:,:,:] = np.expand_dims(query_mask[start_num:total_proposal,:,:],axis=3)
                proposal_transfer_data = sess.run(proposal_transfer, feed_dict={label: np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=1), axis=1)),axis=1).transpose([0, 2, 3, 1]), proposal: proposal_val, mask: proposal_mask})
                proposal_transfered_result[start_num:total_proposal, :, :,:] = proposal_transfer_data[0:residual_num, :, :, :]
            else:
                start_num = n*num_proposal
                end_num = (n+1)*num_proposal

                proposal_val = proposals[start_num:end_num,:,:,:]
                proposal_mask = np.expand_dims(query_mask[start_num:end_num,:,:],axis=3)
                proposal_transfer_data = sess.run(proposal_transfer, feed_dict={label: np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=1), axis=1)),axis=1).transpose([0, 2, 3, 1]), proposal: proposal_val, mask: proposal_mask})

                proposal_transfered_result[start_num:end_num, :, :, :] = proposal_transfer_data
        proposal_transfered_result = np.uint8(proposal_transfered_result*255.0)
        scipy.io.savemat(save_result_path+"/%08d.mat"%(ind),{'transferred':proposal_transfered_result})


