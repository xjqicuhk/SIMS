from __future__ import division
import os,cv2,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from spatial_transformer import transformer
import numpy as np
import numpy.matlib


training_phase = False
if(training_phase):
    label_path = "../traindata/label_refine/"
    mat_path  = "../traindata/order/"
    save_model_path = "../trainedmodels/order"
else:
    label_path = "../testdata/label_refine/"
    mat_path = '../testdata/order/'
    save_result_path ="../result/order/data/"
    save_model_path = "../trainedmodels/order"

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')


batch_size = 10
crop_size_h = 256
crop_size_w = 512
input_dim = 6
sp = 256
topk = 10

sess=tf.Session()


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
def one_hot(label):
    #print(label.shape)
    output=np.zeros((1,100,label.shape[0],label.shape[1]),dtype=np.float32);
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]>0:
                output[0,label[i,j]-1,i,j]=1
    return output



def proposal_classifier(data):
    conv1_selection = slim.repeat(data, 2, slim.conv2d, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_selection_conv1')
    pool1_selection = slim.avg_pool2d(conv1_selection, [3, 3], stride=2, padding='SAME', scope='g_selection_pool1')

    conv2_selection = slim.repeat(pool1_selection, 2, slim.conv2d, 128, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_selection_conv2')
    pool2_selection = slim.max_pool2d(conv2_selection, [3, 3], stride=2, padding='SAME', scope='g_selection_pool2')

    conv3_selection = slim.repeat(pool2_selection, 3, slim.conv2d, 256, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_selection_conv3')
    pool3_selection = slim.max_pool2d(conv3_selection, [3, 3], stride=2, padding='SAME', scope='g_selection_pool3')

    conv4_selection = slim.repeat(pool3_selection, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_selection_conv4')
    pool4_selection = slim.max_pool2d(conv4_selection, [3, 3], stride=2, padding='SAME', scope='g_selection_pool4')

    conv5_selection = slim.repeat(pool4_selection, 3, slim.conv2d, 512, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu,
                                  scope='g_selection_conv5')
    pool5_selection = slim.max_pool2d(conv5_selection, [3, 3], stride=2, padding='SAME', scope='g_selection_pool5')

    pool6_selection = slim.avg_pool2d(pool5_selection, [8, 12], stride=[8, 12], padding='VALID',
                                      scope='g_selection_pool6')
    fc_selection = slim.conv2d(pool6_selection, 1024, [1, 1], rate=1, normalizer_fn=slim.layer_norm,
                                  activation_fn=lrelu, scope='g_selection_conv6')
    fc_selection = slim.dropout(fc_selection, 0.5, is_training= training_phase, scope='drop7_selection_norm')

    score = slim.conv2d(fc_selection, 20, [1, 1], rate=1,
                                  activation_fn=None, scope='g_selection_classify')
    score = tf.squeeze(score)

    return score



with tf.variable_scope(tf.get_variable_scope()) as scope:
    label=tf.placeholder(tf.int32,[batch_size,1])
    data=tf.placeholder(tf.float32,[batch_size,None,None,input_dim])
    sp=256
    proposal_score = proposal_classifier(data)

    weight=tf.placeholder(tf.float32)

    label = tf.squeeze(label)
    G_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits= proposal_score)
    G_loss = tf.reduce_mean(G_loss)



lr=tf.placeholder(tf.float32)

t_vars = tf.trainable_variables()
for var in t_vars:
    if var.name.startswith('g_'):
        print var.name

G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss = G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(save_model_path+ "/0100")
if ckpt:
   print('loaded '+ckpt.model_checkpoint_path)
   saver.restore(sess,ckpt.model_checkpoint_path)


best=np.inf

g_loss=np.zeros(3000,dtype=float)

tmp_label = np.zeros((batch_size,1),dtype = np.int32)
tmp_data = np.zeros((batch_size, crop_size_h,crop_size_w, input_dim), dtype = float)
tmp_mask = np.zeros((batch_size,1), dtype =float)

total_number = 2975
base_lr = 1e-6
if(training_phase):
    for epoch in range(1,101):
        tmp_label = np.zeros((batch_size,1),dtype = np.int32)
        tmp_data = np.zeros((batch_size, crop_size_h,crop_size_w, input_dim), dtype = float)
        tmp_mask = np.zeros((batch_size,1), dtype =float)
        if os.path.isdir(save_model_path+"/%04d"%epoch):
            continue

        tmp_list = np.random.permutation(total_number) + 1
        global_ind = 0
        while(global_ind<total_number):
            st=time.time()

            try:
                dic = scipy.io.loadmat(mat_path + "%08d.mat" % tmp_list[global_ind])
            except:
                print("cannot load"+ "%08d.mat"%tmp_list[global_ind])
                global_ind=global_ind + 1
                continue

            topk_proposal = dic['semantic_segment_mask'].astype(np.float32)
            if (topk_proposal.ndim == 0):
                global_ind = global_ind + 1

                continue

            corr_label = dic['semantic_segment_label'].astype(np.int32)
            topk_proposal = topk_proposal/255.0



            semantic = cv2.imread(label_path+"/%08d.png"%tmp_list[global_ind]).astype(np.float32)
            semantic = semantic/255.0
            tmp_semantic = semantic.copy()
            tmp_semantic[:,:,0] = semantic[:,:,2].copy()
            tmp_semantic[:,:,1] = semantic[:,:,1].copy()
            tmp_semantic[:,:,2] = semantic[:,:,0].copy()

            if(topk_proposal.ndim==3):
                tmp_data[0,:,:,0:3] = topk_proposal
                tmp_data[0,:,:,3:6] = tmp_semantic
                tmp_label[0,:] = corr_label

            if(topk_proposal.ndim == 4):
               num_proposal = topk_proposal.shape[3]
               if(num_proposal>=batch_size):
                   tmp_data[0:batch_size,:,:,0:3] = np.transpose(topk_proposal[:,:,:,0:batch_size],[3,0,1,2])
                   tmp_data[0:batch_size,:,:,3:6] = np.tile(np.expand_dims(tmp_semantic,axis=0),(batch_size,1,1,1))
                   tmp_label[0:batch_size,:] = corr_label[0:batch_size,:]
               else:
                   tmp_data[0:num_proposal, :, :, 0:3] = np.transpose(topk_proposal[:, :, :, 0:num_proposal], [3, 0,1,2])
                   tmp_data[0:num_proposal, :, :, 3:6] = np.tile(np.expand_dims(tmp_semantic,axis=0),(num_proposal,1,1,1))
                   tmp_label[0:num_proposal, :] = corr_label[0:num_proposal, :]
            print(tmp_label[0])


            _,G_current,lr_value, proposal_score_val=sess.run([G_opt,G_loss,lr,proposal_score],feed_dict={label:np.squeeze(tmp_label),data:tmp_data,lr:min(base_lr*np.power(1.1,epoch-1),1e-5 )})

            g_loss[global_ind]=G_current
            global_ind = global_ind + 1

            print("%d %d %.2f %.2f %.2f %.6f"%(epoch,global_ind,G_current,np.mean(g_loss[np.where(g_loss)]),time.time()-st, lr_value))


        if epoch%1==0:
            os.makedirs(save_model_path+"/%04d" % epoch)
            target = open(save_model_path+"/%04d/score.txt" % epoch, 'w')
            target.write("%f" % np.mean(g_loss[np.where(g_loss)]))
            target.close()
            saver.save(sess,save_model_path+"/model.ckpt")
        if epoch%20==0:

            saver.save(sess,save_model_path+"/%04d/model.ckpt"%epoch)

else:
    if not os.path.isdir(save_result_path):
        os.makedirs(save_result_path)
    for ind in range(100001,100501):

        print(ind)
        dic = scipy.io.loadmat(mat_path + "%08d.mat" %ind)

        proposal_all = dic['semantic_segment_mask'].astype(np.float32)/255.0

        if (proposal_all.shape[0] == 0):
            continue
        if (proposal_all.ndim == 3 ):
            print('expanding....')
            proposal_all = np.expand_dims(proposal_all, axis=3)

        semantic = cv2.imread(label_path+"/%08d.png" % ind)
        semantic = semantic.astype(np.float32) / 255.0
        tmp_semantic = semantic.copy()
        tmp_semantic[:, :, 0] = semantic[:, :, 2].copy()
        tmp_semantic[:, :, 1] = semantic[:, :, 1].copy()
        tmp_semantic[:, :, 2] = semantic[:, :, 0].copy()


        final_pred = np.zeros((proposal_all.shape[3],20), dtype=np.float32)
        count_all = proposal_all.shape[3]
        index = 0

        while(count_all>0):
            tmp_data = np.zeros((batch_size, crop_size_h, crop_size_w, input_dim), dtype=float)
            if(count_all>batch_size):
                tmp_data[0:batch_size,:,:,0:3] = np.transpose(proposal_all[:,:,:,index:index+batch_size],[3,0,1,2])
                tmp_data[0:batch_size,:,:,3:6] = np.tile(np.expand_dims(tmp_semantic,axis=0),(batch_size,1,1,1))
                final_pred[index:index+batch_size,: ] = np.squeeze(sess.run(proposal_score, feed_dict={data: tmp_data}))
                count_all = count_all - batch_size
                index = index+batch_size
            else:
                tmp_data[0:count_all, :, :, 0:3] = np.transpose(proposal_all[:, :, :, index:index + count_all],[3, 0, 1, 2])
                tmp_data[0:count_all, :, :, 3:6] = np.tile(np.expand_dims(tmp_semantic, axis=0), (count_all, 1, 1, 1))

                final_pred[index:index+count_all,:] = np.squeeze(sess.run(proposal_score, feed_dict={data: tmp_data}))[0:count_all,:]
                index = index + count_all
                count_all = 0

        scipy.io.savemat(save_result_path + "/%08d.mat" % ind, {'prediction': final_pred})


