# testing phase
'''
 testing phase:
 input: canvas + label map
 output: synthesized image
 training phase:
 input: cruppted canvas + label map
 Ground truth: original rgb image and original label map
'''

from __future__ import division
import os,cv2,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np


training_phase = True

if(training_phase):
    rgb_path = "../traindata/RGB256Full/"
    mat_path = "../traindata/synthesis/traindata_synthesis_256_512/traindata_mat/"
    label_path = "../traindata/synthesis/traindata_synthesis_256_512/traindata_label/"
    label_gt_path = "../traindata/Label256Full/"
    save_model_path = "../trainedmodels/synthesis_256_512"
    num_folder = 5
else:
    save_model_path = "../trainedmodels/synthesis_256_512"
    mat_path = "../testdata/synthesis/transform_order_256/"
    save_result_path = "../result/synthesis/transform_order_256/"


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

batch_size = 1
# resolution height and width
crop_size_h = 256
crop_size_w = 512
# resolution
sp = 256
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

def recursive_generator(label,sp):

    conv1_encoder = slim.repeat(label, 2, slim.conv2d, 64, [3, 3], rate=1,
                                  normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_encoder_conv1')
    pool1_encoder = slim.avg_pool2d(conv1_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool1')

    conv2_encoder = slim.repeat(pool1_encoder, 2, slim.conv2d, 128, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_encoder_conv2')
    pool2_encoder = slim.avg_pool2d(conv2_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool2')

    conv3_encoder = slim.repeat(pool2_encoder, 3, slim.conv2d, 256, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_encoder_conv3')
    pool3_encoder = slim.avg_pool2d(conv3_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool3')

    conv4_encoder = slim.repeat(pool3_encoder, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_encoder_conv4')
    pool4_encoder = slim.avg_pool2d(conv4_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool4')

    conv5_encoder = slim.repeat(pool4_encoder, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_encoder_conv5')
    pool5_encoder = slim.avg_pool2d(conv5_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool5')

    conv6_encoder = slim.repeat(pool5_encoder, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                activation_fn=lrelu,
                                scope='g_encoder_conv6')
    pool6_encoder = slim.avg_pool2d(conv6_encoder, [3, 3], stride=2, padding='SAME', scope='g_encoder_pool6')

    #decoder
    downsampled_6 = tf.image.resize_bilinear(label, (sp//64, sp//32), align_corners=True)
    input_6 = tf.concat([downsampled_6,pool6_encoder],3)
    net_6 = slim.repeat(input_6, 2, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu,
                                scope='g_decoder_conv6')
    net_6 = tf.image.resize_bilinear(net_6,(sp//32,sp//16), align_corners=True)


    downsampled_5 = tf.image.resize_bilinear(label, (sp//32, sp//16), align_corners=True)
    input_5 = tf.concat([downsampled_5, pool5_encoder,net_6], 3)
    net_5 = slim.repeat(input_5, 2, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu,
                        scope='g_decoder_conv5')
    net_5 = tf.image.resize_bilinear(net_5, (sp // 16, sp // 8), align_corners=True)

    downsampled_4 = tf.image.resize_bilinear(label, (sp // 16, sp // 8), align_corners=True)
    input_4 = tf.concat([downsampled_4, pool4_encoder, net_5], 3)
    net_4 = slim.repeat(input_4,2, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu,
                        scope='g_decoder_conv4')
    net_4= tf.image.resize_bilinear(net_4, (sp // 8, sp // 4), align_corners=True)


    downsampled_3 = tf.image.resize_bilinear(label, (sp // 8, sp // 4), align_corners=True)
    input_3 = tf.concat([downsampled_3, pool3_encoder, net_4], 3)
    net_3 = slim.repeat(input_3, 2, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu,
                        scope='g_decoder_conv3')
    net_3 = tf.image.resize_bilinear(net_3, (sp // 4, sp // 2), align_corners=True)

    downsampled_2 = tf.image.resize_bilinear(label, (sp // 4, sp // 2), align_corners=True)
    input_2 = tf.concat([downsampled_2, pool2_encoder, net_3], 3)
    net_2 = slim.repeat(input_2, 2, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu,
                        scope='g_decoder_conv2')
    net_2 = tf.image.resize_bilinear(net_2, (sp // 2, sp), align_corners=True)

    downsampled_1 = tf.image.resize_bilinear(label, (sp // 2,sp), align_corners=True)
    input_1 = tf.concat([downsampled_1, pool1_encoder, net_2], 3)
    net_1 = slim.repeat(input_1, 2, slim.conv2d, 256, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu,
                        scope='g_decoder_conv1')
    net_1 = tf.image.resize_bilinear(net_1, (sp, 2*sp), align_corners=True)

    input = tf.concat([label,net_1],3)
    net = slim.repeat(input, 2, slim.conv2d, 256, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu,
                        scope='g_decoder_final')

    net_semantic = slim.conv2d(net,20,[1,1],rate=1,activation_fn=None, scope= 'g_semantic_'+str(sp)+'_conv100')


    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
    net=(net+1.0)/2.0*255.0


    return net,net_semantic


def compute_error(real,fake,label):
    return tf.reduce_sum(tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[0,3]),-1),reduction_indices=[0,1,2]))




with tf.variable_scope(tf.get_variable_scope()) as scope:
    label=tf.placeholder(tf.float32,[batch_size,None,None,20])
    label_gt = tf.placeholder(tf.float32,[batch_size,None,None,20])
    real_image=tf.placeholder(tf.float32,[batch_size,None,None,3])
    fake_image=tf.placeholder(tf.float32,[batch_size,None,None,3])
    mask = tf.placeholder(tf.float32,[batch_size,crop_size_h,crop_size_w,1])
    proposal = tf.placeholder(tf.float32,[batch_size,crop_size_h,crop_size_w,3])

    proposal_new = proposal
    label_input = tf.concat([label,proposal_new,mask],axis=3)
    #generate the network structure
    generator,generator_semantic=recursive_generator(label_input,sp)

    p_semantic = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(label_gt,[batch_size*crop_size_h*crop_size_w,20]), logits = tf.reshape(generator_semantic,[batch_size*crop_size_h*crop_size_w,20])))

    weight=tf.placeholder(tf.float32)
    #forward vgg networks
    vgg_real=build_vgg19(real_image)
    vgg_fake=build_vgg19(generator,reuse=True)
    # calculate perceptron losses + minimizing the 1 vs all loss.
    p0=compute_error(vgg_real['input'],vgg_fake['input'],label)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label)/2.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_bilinear(label,(sp//2,sp), align_corners = True))/4.8
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_bilinear(label,(sp//4,sp//2),align_corners = True))/3.7
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_bilinear(label,(sp//8,sp//4),align_corners = True))/5.6
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_bilinear(label,(sp//16,sp//8),align_corners = True))*10/1.5
    G_loss=p0+p1+p2+p3+p4+p5+p_semantic*10



lr=tf.placeholder(tf.float32)


t_vars = tf.trainable_variables()
for var in t_vars:
    if var.name.startswith('g_'):
        print var.name
 
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss = G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(save_model_path+ "/0800")
if ckpt:
   print('loaded '+ckpt.model_checkpoint_path)
   saver.restore(sess,ckpt.model_checkpoint_path)


best=np.inf
all_loss=np.zeros(3000,dtype=float)
g_loss=np.zeros(3000,dtype=float)
all_c_loss=np.zeros(3000,dtype=float)
input_images=[None]*3000
label_images=[None]*3000
if training_phase:
    tmp_label = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
    tmp_label_gt = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
    tmp_image = np.zeros((batch_size, crop_size_h,crop_size_w, 3), dtype = float)
    tmp_mask = np.zeros((batch_size,crop_size_h, crop_size_w,1), dtype =float)
    tmp_proposal = np.zeros((batch_size,crop_size_h,crop_size_w,3),dtype = float)
    total_number = 2975
    base_lr = 1e-6
    for epoch in range(1,801):
        if os.path.isdir(save_path+ file_folder+"/%04d"%epoch):
            continue
        cnt=0

        tmp_list = np.random.permutation(total_number) + 1
        global_ind = 0
        ind = 0
        while(global_ind<total_number):
            st=time.time()
            for batch_count in range(0, batch_size):
                if(ind>=total_number):
                    ind = 0
                cnt += 1
                rand_folder = np.random.permutation(num_folder) + 1

                try:
                    dic = scipy.io.loadmat(mat_path +"%02d"%rand_folder[0]+"/%08d.mat" % tmp_list[ind])
                except:
                    print("cannot load" + "%08d.mat" % tmp_list[ind])
                    print(mat_path +"%02d"%rand_folder[0]+"/%08d.mat" % tmp_list[ind])
                    ind = ind + 1
                    global_ind = global_ind + 1
                    continue
                tmp_label[batch_count, :, :, :] = helper.get_semantic_map(
                    label_path + "/%02d" % rand_folder[0] + "/%08d.png" % tmp_list[ind])
                tmp_image[batch_count, :, :, :] = np.expand_dims(
                    np.float32(cv2.imread(rgb_path+"/%08d.png" % tmp_list[ind])), axis=0)
                tmp_label_gt[batch_count, :,:,:] =  helper.get_semantic_map(
                    data_path + label_gt_folder + "/%08d.png" % tmp_list[ind])

                tmp_proposal[batch_count,:,:,:] = (np.expand_dims(dic['proposal'],axis = 0).astype(np.float32))/255.0#+ np.expand_dims(dic2['proposal'],axis = 0)

                tmp_z = np.sum(tmp_proposal[batch_count, :, :, :], axis=-1)
                tmp_z[np.where(tmp_z > 0)] = 1
                tmp_mask[batch_count, :, :, :] = np.expand_dims(tmp_z, axis=3).astype(np.float32)


                z = np.concatenate((tmp_label, np.expand_dims(1 - np.sum(tmp_label, axis=1), axis=1)), axis=1)
                z = z.transpose([0,2,3,1])

                z_gt = np.concatenate((tmp_label_gt, np.expand_dims(1 - np.sum(tmp_label_gt, axis=1), axis=1)), axis=1)
                z_gt = z_gt.transpose([0,2,3,1])

                ind = ind + 1
                global_ind = global_ind + 1
            _,G_current,l0,l1,l2,l3,l4,l5,p_semantic_value, lr_value=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5,p_semantic, lr],feed_dict={label:z,real_image:tmp_image,mask: tmp_mask, label_gt: z_gt, proposal:tmp_proposal,lr:min(base_lr*np.power(1.1,epoch-1),1e-4 if epoch>112 else 1e-3)})
            g_loss[cnt]=G_current
            print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.6f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),p_semantic_value,time.time()-st, lr_value))


        if epoch%1==0:
            os.makedirs(save_model_path+"/%04d" % epoch)
            target = open(save_model_path+"/%04d/score.txt" % epoch, 'w')
            target.write("%f" % np.mean(g_loss[np.where(g_loss)]))
            target.close()
            saver.save(sess,save_model_path+"/model.ckpt")
        if epoch%20==0:
            saver.save(sess,save_model_path+"/%04d/model.ckpt"%epoch)
        cnt=3000

        # validation list
        tmp_list = range(2800,2976)
        global_count = 0
        semantic = np.zeros((batch_size,19,crop_size_h,crop_size_w),dtype = float)
        proposal_val = np.zeros((batch_size,crop_size_h,crop_size_w,3),dtype = float)
        tmp_mask = np.zeros((batch_size, crop_size_h, crop_size_w, 1), dtype=float)
        while (global_count < 100):
            for local_count in range(0,batch_size):
                rand_folder = np.random.permutation(num_folder) + 1
                try:
                    dic = scipy.io.loadmat(mat_path + "%02d"%rand_folder[0]+"/%08d.mat"%tmp_list[global_count])
                except:
                    print("cannot load"+"%08d.mat"%tmp_list[ind])
                    global_count = global_count +1
                semantic[local_count,:,:,:]=helper.get_semantic_map(data_path+label_folder+"/%02d"%rand_folder[0]+"/%08d.png"%tmp_list[global_count])
                proposal_val[local_count,:,:,:] = (np.expand_dims(dic['proposal'],axis = 0).astype(np.float32))/255.0 #+ np.expand_dims(dic2['proposal'],axis = 0)

                tmp_z = np.sum(proposal_val[local_count, :, :, :], axis=-1)
                tmp_z[np.where(tmp_z > 0)] = 1
                tmp_mask[local_count, :, :, :] = np.expand_dims(tmp_z, axis=3).astype(np.float32)

                global_count = global_count +1

            z = np.concatenate((semantic, np.expand_dims(1 - np.sum(semantic, axis=1), axis=1)), axis=1)
            z = z.transpose([0, 2, 3, 1])

            output, output_semantic=sess.run([generator,generator_semantic],feed_dict={label:z,proposal:proposal_val, mask:tmp_mask})
            output=np.minimum(np.maximum(output,0.0),255.0)

            output_semantic = np.squeeze(output_semantic)
            final = np.argmax(output_semantic, axis=-1)

            cv2.imwrite(save_path+ file_folder+"/%04d/%06d_output.jpg"%(epoch,cnt),np.uint8(output[0,:,:,:]))
            scipy.io.savemat(save_path + file_folder + "/%04d/%06d_output.jpg" % (epoch, cnt), {'label_pred': final})
            cnt = cnt + batch_size
else:
    if not os.path.isdir(save_result_path):
        os.makedirs(save_result_path)
    for ind in range(100001,100501):
        print(ind)
        semantic=helper.get_semantic_map(label_folder+"/%08d.png"%ind)
        dic = scipy.io.loadmat(mat_path + "%08d.mat" %ind)
        proposal_tmp = np.expand_dims(dic['proposal'], axis=0)
        tmp_z = np.sum(proposal_tmp, axis=-1)
        tmp_z[np.where(tmp_z > 0)] = 1
        tmp_mask = np.expand_dims(tmp_z, axis=3).astype(np.float32)
        semantic = helper.encode_semantic_map(dic['label'])

        output,output_semantic=sess.run([generator,generator_semantic],feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=1),axis=1)),axis=1).transpose([0,2,3,1]),proposal:proposal_tmp,mask:tmp_mask})
        output=np.minimum(np.maximum(output, 0.0), 255.0)
        output_semantic = np.squeeze(output_semantic)
        final = np.argmax(output_semantic, axis=-1)
        cv2.imwrite(save_result_path+"/%08d.png" %ind,np.uint8(output[0, :, :, :]))

