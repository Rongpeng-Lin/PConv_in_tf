import numpy as np
import scipy.misc as misc
import math
import vgg
import os

from cacu_loss import *
from get_input import *
from ops import *

class Image_inpainting:
    def __init__(self,im_path,vgg_path,num_epoch,logdir,save_path):       
        self.ims_dir = [im_path+im_name for im_name in os.listdir(im_path)]
        self.total_ims = len(self.ims_dir)
        self.vgg_path = vgg_path
        self.logdir = logdir
        self.save_path = save_path
        self.batch = 1
        self.num_epochs = num_epoch
        self.vgg_layer = ['pool1','pool2','pool3']
        
        holder_names = ['mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4', 'mask_conv5', 'mask_conv6', 'mask_conv7', 'mask_conv8', 'mask_merge_convd0', 'mask_merge_convd1', 'mask_merge_convd2', 'mask_merge_convd3', 'mask_merge_convd4', 'mask_merge_convd5', 'mask_merge_convd6', 'mask_merge_convd7', 'mask_original', 'mask_up0', 'mask_up1', 'mask_up2', 'mask_up3', 'mask_up4', 'mask_up5', 'mask_up6', 'mask_up7', 'mask_up_new0', 'mask_up_new1', 'mask_up_new2', 'mask_up_new3', 'mask_up_new4', 'mask_up_new5', 'mask_up_new6', 'mask_up_new7', 'new_mask1', 'new_mask2', 'new_mask3', 'new_mask4', 'new_mask5', 'new_mask6', 'new_mask7', 'new_mask8', 'image_mask', 'image_gt']
        holder_shapes = [[1, 256, 256, 1], [1, 128, 128, 1], [1, 64, 64, 1], [1, 32, 32, 1], [1, 16, 16, 1], [1, 8, 8, 1], [1, 4, 4, 1], [1, 2, 2, 1], [1, 512, 512, 1], [1, 256, 256, 1], [1, 128, 128, 1], [1, 64, 64, 1], [1, 32, 32, 1], [1, 16, 16, 1], [1, 8, 8, 1], [1, 4, 4, 1], [1, 512, 512, 1], [1, 512, 512, 1], [1, 256, 256, 1], [1, 128, 128, 1], [1, 64, 64, 1], [1, 32, 32, 1], [1, 16, 16, 1], [1, 8, 8, 1], [1, 4, 4, 1], [1, 512, 512, 1], [1, 256, 256, 1], [1, 128, 128, 1], [1, 64, 64, 1], [1, 32, 32, 1], [1, 16, 16, 1], [1, 8, 8, 1], [1, 4, 4, 1], [1, 256, 256, 1], [1, 128, 128, 1], [1, 64, 64, 1], [1, 32, 32, 1], [1, 16, 16, 1], [1, 8, 8, 1], [1, 4, 4, 1], [1, 2, 2, 1], [1,512,512,3], [1,512,512,3]]
        holder_dtype = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        self.holder = list(map(tf.placeholder,holder_dtype,holder_shapes,holder_names))
        
    def get_original_mask(self,im):
        mask_init = np.ones_like(im[0,:,:,0],np.float32)
        for i in range(np.shape(im)[1]):
            for j in range(np.shape(im)[2]):
                if (im[0,i,j,0]==0) and (im[0,i,j,1]==0) and (im[0,i,j,2]==0):
                        mask_init[i,j] = 0.
        return mask_init[None,:,:,None]  
     
    def mask_conv(self,mask,ker_size,pad,s,input_cha):  ### mask 是四维的
        mask = mask[0,:,:,0]
        mask_pad = np.zeros([np.shape(mask)[0]+2*pad,np.shape(mask)[0]+2*pad])
        mask_pad[pad:pad+np.shape(mask)[0],pad:pad+np.shape(mask)[0]] = mask    
        ker_l = int(math.sqrt(ker_size))
        kernel = np.ones([ker_l,ker_l],np.float32)
        ker_sum = np.sum(kernel)*input_cha
        out_shape = 1+(np.shape(mask)[0]-int(math.sqrt(ker_size))+2*pad)//s   
        mask_conved = np.zeros([out_shape,out_shape],np.float32)
        for i in range(0,out_shape):
            for j in range(0,out_shape):
                su = np.sum(mask_pad[int(s*i):int(s*i+ker_l),int(s*j):int(s*j+ker_l)]*kernel)            
                if su!=0.:
                    mask_conved[i,j] = ker_sum/(su*input_cha) 
        new_mask = np.equal(np.equal(mask_conved,0.),0).astype(np.float32) 
        return new_mask[None,:,:,None],mask_conved[None,:,:,None]
    
    def two_mask_conv(self,mask,input_cha,mask_ano,input_cha_ano,ker_size,pad,s):  ### mask 是四维的
        mask = mask[0,:,:,0]
        mask_pad = np.zeros([np.shape(mask)[0]+2*pad,np.shape(mask)[0]+2*pad])
        mask_pad[pad:pad+np.shape(mask)[0],pad:pad+np.shape(mask)[0]] = mask    
        ker_l = int(math.sqrt(ker_size))
        kernel = np.ones([ker_l,ker_l],np.float32)
        ker_sum = np.sum(kernel)*(input_cha+input_cha_ano)  
        out_shape = 1+(np.shape(mask)[0]-int(math.sqrt(ker_size))+2*pad)//s   
        mask_conved = np.zeros([out_shape,out_shape],np.float32)
        for i in range(0,out_shape):
            for j in range(0,out_shape):
                su = np.sum(mask_pad[int(s*i):int(s*i+ker_l),int(s*j):int(s*j+ker_l)]*kernel)            
                if su!=0.:
                    mask_conved[i,j] = su*input_cha
        mask_ano = mask_ano[0,:,:,0]
        mask_ano_pad = np.zeros([np.shape(mask_ano)[0]+2*pad,np.shape(mask_ano)[0]+2*pad])     
        mask_ano_pad[pad:pad+np.shape(mask_ano)[0],pad:pad+np.shape(mask_ano)[0]] = mask_ano
        mask_ano_conved = np.zeros([out_shape,out_shape],np.float32)
        for i in range(0,out_shape):
            for j in range(0,out_shape):
                su = np.sum(mask_ano_pad[int(s*i):int(s*i+ker_l),int(s*j):int(s*j+ker_l)]*kernel)            
                if su!=0.:
                    mask_ano_conved[i,j] = su*input_cha_ano
        mask_merge_convd = mask_conved + mask_ano_conved
        for i in range(0,out_shape):
            for j in range(0,out_shape):                       
                if mask_merge_convd[i,j]!=0.:
                    mask_merge_convd[i,j] = ker_sum/mask_merge_convd[i,j]
        mask_merge_newbin = np.equal(np.equal(mask_merge_convd,0.),0).astype(np.float32)
        return mask_merge_newbin[None,:,:,None],mask_merge_convd[None,:,:,None] 
    
    def mask_upsample(self,mask,want_shape):
        msk = mask[0,:,:,0]
        sh = int(math.sqrt(want_shape))
        up_mask = np.zeros([sh,sh],np.float32)
        for i in range(sh):
            for j in range(sh):
                up_mask[i,j] = msk[round(i*(np.shape(msk)[0]-1)/(sh-1)),round(j*(np.shape(msk)[1]-1)/(sh-1))]          
        return up_mask[None,:,:,None]

    def get_all_mask(self,with_mask,with_gt):
        Mask_original = self.get_original_mask(with_mask)
        
        new_Mask1,Mask_conv1 = self.mask_conv(Mask_original,7*7,3,2,3)
        new_Mask2,Mask_conv2 = self.mask_conv(new_Mask1,5*5,2,2,64)
        new_Mask3,Mask_conv3 = self.mask_conv(new_Mask2,3*3,1,2,128)
        new_Mask4,Mask_conv4 = self.mask_conv(new_Mask3,3*3,1,2,256)
        new_Mask5,Mask_conv5 = self.mask_conv(new_Mask4,3*3,1,2,512)
        new_Mask6,Mask_conv6 = self.mask_conv(new_Mask5,3*3,1,2,512)
        new_Mask7,Mask_conv7 = self.mask_conv(new_Mask6,3*3,1,2,512)
        new_Mask8,Mask_conv8 = self.mask_conv(new_Mask7,3*3,1,2,512)
        
        Mask_up7 = self.mask_upsample(new_Mask8,4*4)
        Mask_up_new7,Mask_merge_convd7 = self.two_mask_conv(Mask_up7,512,new_Mask7,512,3*3,1,1)

        Mask_up6 = self.mask_upsample(Mask_up_new7,8*8)        
        Mask_up_new6,Mask_merge_convd6 = self.two_mask_conv(Mask_up6,512,new_Mask6,512,3*3,1,1)

        Mask_up5 = self.mask_upsample(Mask_up_new6,16*16)
        Mask_up_new5,Mask_merge_convd5 = self.two_mask_conv(Mask_up5,512,new_Mask5,512,3*3,1,1)

        Mask_up4 = self.mask_upsample(Mask_up_new5,32*32)
        Mask_up_new4,Mask_merge_convd4 = self.two_mask_conv(Mask_up4,512,new_Mask4,512,3*3,1,1)

        Mask_up3 = self.mask_upsample(Mask_up_new4,64*64)
        Mask_up_new3,Mask_merge_convd3 = self.two_mask_conv(Mask_up3,512,new_Mask3,256,3*3,1,1)

        Mask_up2 = self.mask_upsample(Mask_up_new3,128*128)
        Mask_up_new2,Mask_merge_convd2 = self.two_mask_conv(Mask_up2,256,new_Mask2,128,3*3,1,1)

        Mask_up1 = self.mask_upsample(Mask_up_new2,256*256)
        Mask_up_new1,Mask_merge_convd1 = self.two_mask_conv(Mask_up1,128,new_Mask1,64,3*3,1,1)

        Mask_up0 = self.mask_upsample(Mask_up_new1,512*512)
        Mask_up_new0,Mask_merge_convd0 = self.two_mask_conv(Mask_up0,64,Mask_original,3,3*3,1,1)
        self.all_masks = [Mask_conv1,Mask_conv2,
                          Mask_conv3,Mask_conv4,
                          Mask_conv5,Mask_conv6,
                          Mask_conv7,Mask_conv8,
                          Mask_merge_convd0,Mask_merge_convd1,
                          Mask_merge_convd2,Mask_merge_convd3,
                          Mask_merge_convd4,Mask_merge_convd5,
                          Mask_merge_convd6,Mask_merge_convd7,
                          Mask_original,
                          Mask_up0,Mask_up1,
                          Mask_up2,Mask_up3,
                          Mask_up4,Mask_up5,
                          Mask_up6,Mask_up7,
                          Mask_up_new0,Mask_up_new1,
                          Mask_up_new2,Mask_up_new3,
                          Mask_up_new4,Mask_up_new5,
                          Mask_up_new6,Mask_up_new7,
                          new_Mask1,new_Mask2,
                          new_Mask3,new_Mask4,
                          new_Mask5,new_Mask6,
                          new_Mask7,new_Mask8,
                          with_mask,with_gt]
          
    def U_net(self,x):
        x = (x/127.5)-1
        with tf.variable_scope('UNET'):
            conv1 = conv('conv1',x,7*7,64,2,self.holder[16],self.holder[0],self.holder[33])
            conv1_relu = relu('conv1_relu',conv1)

            conv2 = conv('conv2',conv1_relu,5*5,128,2,self.holder[33],self.holder[1],self.holder[34])
            conv2_bn = ins_norm('conv2_ins',conv2,self.holder[34])
            conv2_relu = relu('conv2_relu',conv2_bn)

            conv3 = conv('conv3',conv2_relu,3*3,256,2,self.holder[34],self.holder[2],self.holder[35])
            conv3_bn = ins_norm('conv3_ins',conv3,self.holder[35])
            conv3_relu = relu('conv3_relu',conv3_bn)

            conv4 = conv('conv4',conv3_relu,3*3,512,2,self.holder[35],self.holder[3],self.holder[36])
            conv4_bn = ins_norm('conv4_ins',conv4,self.holder[36])
            conv4_relu = relu('conv4_relu',conv4_bn)

            conv5 = conv('conv5',conv4_relu,3*3,512,2,self.holder[36],self.holder[4],self.holder[37])
            conv5_bn = ins_norm('conv5_ins',conv5,self.holder[37])
            conv5_relu = relu('conv5_relu',conv5_bn)

            conv6 = conv('conv6',conv5_relu,3*3,512,2,self.holder[37],self.holder[5],self.holder[38])
            conv6_bn = ins_norm('conv6_ins',conv6,self.holder[38])
            conv6_relu = relu('conv6_relu',conv6_bn)

            conv7 = conv('conv7',conv6_relu,3*3,512,2,self.holder[38],self.holder[6],self.holder[39])
            conv7_bn = ins_norm('conv7_ins',conv7,self.holder[32])
            conv7_relu = relu('conv7_relu',conv7_bn)

            conv8 = conv('conv8',conv7_relu,3*3,512,2,self.holder[39],self.holder[7],self.holder[40])
            conv8_relu = relu('conv8_relu',conv8)

            conv_up7 = conv_up('conv_up7', conv8_relu, self.holder[24], conv7_relu, self.holder[39], self.holder[15], self.holder[32], 3*3, 512, 1)                             
            conv_up7bn = ins_norm('conv_up7ins',conv_up7,self.holder[32])
            conv_up7relu = relu('conv_up7relu',conv_up7bn)

            conv_up6 = conv_up('conv_up6', conv_up7relu, self.holder[23], conv6_relu, self.holder[38], self.holder[14], self.holder[31], 3*3, 512, 1)                             
            conv_up6bn = ins_norm('conv_up6ins',conv_up6,self.holder[31])
            conv_up6relu = relu('conv_up6relu',conv_up6bn)

            conv_up5 = conv_up('conv_up5', conv_up6relu, self.holder[22], conv5_relu, self.holder[37], self.holder[13], self.holder[30], 3*3, 512, 1)                             
            conv_up5bn = ins_norm('conv_up5ins',conv_up5,self.holder[30])
            conv_up5relu = relu('conv_up5relu',conv_up5bn)

            conv_up4 = conv_up('conv_up4', conv_up5relu, self.holder[21], conv4_relu, self.holder[36], self.holder[12], self.holder[29], 3*3, 512, 1)                             
            conv_up4bn = ins_norm('conv_up4ins',conv_up4,self.holder[29])
            conv_up4relu = relu('conv_up4relu',conv_up4bn)

            conv_up3 = conv_up('conv_up3', conv_up4relu, self.holder[20], conv3_relu, self.holder[35], self.holder[11], self.holder[28], 3*3, 256, 1)                             
            conv_up3bn = ins_norm('conv_up3ins',conv_up3,self.holder[28])
            conv_up3relu = relu('conv_up3relu',conv_up3bn)

            conv_up2 = conv_up('conv_up2', conv_up3relu, self.holder[19], conv2_relu, self.holder[34], self.holder[10], self.holder[27], 3*3, 128, 1)                             
            conv_up2bn = ins_norm('conv_up2ins',conv_up2,self.holder[27])
            conv_up2relu = relu('conv_up2relu',conv_up2bn)

            conv_up1 = conv_up('conv_up1', conv_up2relu, self.holder[18], conv1_relu, self.holder[33], self.holder[9], self.holder[26], 3*3, 64, 1)                             
            conv_up1bn = ins_norm('conv_up1ins',conv_up1,self.holder[26])
            conv_up1relu = relu('conv_up1relu',conv_up1bn)

            conv_up0 = conv_up('conv_up0', conv_up1relu, self.holder[17], x, self.holder[16], self.holder[8], self.holder[25], 3*3, 3, 1)                             
            conv_up0tanh = tanh('conv_up0relu',conv_up0)
            return conv_up0tanh
        
    def train(self):
        with tf.Session() as sess:
            out_im = self.U_net(self.holder[41]/127.5-1)
            
            gt_resize = tf.image.resize_images(self.holder[42]/127.5-1, [256,256])
            image_pre = vgg.preprocess(gt_resize)
            fai_imgt = {}
            net = vgg.net(self.vgg_path, image_pre)
            for layer in self.vgg_layer:
                fai_imgt[layer] = net[layer]
                
            
            image_pre = vgg.preprocess(tf.image.resize_images(out_im, [256,256]))
            fai_imout = {}
            net = vgg.net(self.vgg_path, image_pre)
            for layer in self.vgg_layer:
                fai_imout[layer] = net[layer]
            
            Im_compt = self.holder[16]*self.holder[42]+(tf.add(tf.multiply(self.holder[16],-1),1))*((out_im+1)*127.5)
            im_compt = tf.image.resize_images(Im_compt/127.5-1, [256,256])
            image_pre = vgg.preprocess(im_compt)
            fai_compt = {}
            net = vgg.net(self.vgg_path, image_pre)
            for layer in self.vgg_layer:
                fai_compt[layer] = net[layer]
                
            U_vars = [var for var in tf.trainable_variables() if 'UNET' in var.name]
            total_loss = get_total_loss(out_im,self.holder[-1]/127.5-1,self.holder[16],fai_imout,fai_imgt,fai_compt,self.vgg_layer,im_compt)
            optim = tf.train.AdamOptimizer()
            optimizer = optim.minimize(total_loss[0],var_list=U_vars)
            
    
            int_group = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(int_group)
            
            graph = tf.summary.FileWriter(self.logdir, sess.graph)
            saver = tf.train.Saver(U_vars,max_to_keep=20)
            
            for epoch in range(self.num_epochs):
                for imid in range(int(self.total_ims//self.batch)):
                    mask_ims,gt_ims = get_im(self.ims_dir,imid)
                    self.get_all_mask(mask_ims,gt_ims)
                    feed_dic = get_feedict(self.all_masks,self.holder)
                    _,loss_total = sess.run([optimizer,total_loss],feed_dict=feed_dic)
                    
                    if (int(epoch*self.total_ims)+imid)%1==0:
                        print('epoch: %d,  cur_num: %d,  total_loss: %f, l_hole: %f, l_valid: %f, percept_loss: %f, style_loss_out: %f, style_loss_comp: %f, tv_loss: %f'%(epoch,imid,loss_total[0],loss_total[1],loss_total[2],loss_total[3],loss_total[4],loss_total[5],loss_total[6]))
                if epoch%5==0:
                    saver.save(sess, self.save_path+'model.ckpt', global_step=epoch)
               

