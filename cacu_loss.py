###  out_im: 4_dim trnsor   ####   cacu_loss
###  raw_im: 4_dim trnsor
###  bin_mask: 2_dim array
import tensorflow as tf
import functools

def get_gram(x):
    ba,hi,wi,ch = [i.value for i in x.get_shape()]
    feature = tf.reshape(x,[ba,int(hi*wi),ch])
    feature_T = tf.transpose(feature,[0,2,1])
    gram = tf.matmul(feature_T,feature)
    size = 1/(hi*wi*ch)
    return gram*size

def hole_loss(I_out,I_gt,bin_mask):
    L_hole = tf.reduce_mean(tf.abs((1-bin_mask)*(I_out-I_gt)))
    return L_hole

def valid_loss(I_out,I_gt,bin_mask):
    L_valid = tf.reduce_mean(tf.abs(bin_mask*(I_out-I_gt)))
    return L_valid

def Percept_loss(fai_out,fai_gt,fai_comp,layers):
    out_gt = []
    compt_gt = []
    for layer in layers:
        out_gt.append(tf.reduce_mean(tf.abs(fai_out[layer]-fai_gt[layer])))
        compt_gt.append(tf.reduce_mean(tf.abs(fai_comp[layer]-fai_gt[layer])))
    out_gt_loss = functools.reduce(tf.add,out_gt)
    compt_gt_loss = functools.reduce(tf.add,compt_gt)
    return out_gt_loss+compt_gt_loss

def Style_loss_out(fai_out,fai_gt,layers):
    styleloss = []
    for layer in layers:
        gram_out = get_gram(fai_out[layer])
        gram_gt = get_gram(fai_gt[layer])
        styleloss.append(tf.reduce_mean(tf.abs(gram_out-gram_gt)))
    style_out_loss = functools.reduce(tf.add,styleloss)
    return style_out_loss

def Style_loss_comp(fai_comp,fai_gt,layers):
    styleloss = []
    for layer in layers:
        gram_comp = get_gram(fai_comp[layer])
        gram_gt = get_gram(fai_gt[layer])
        styleloss.append(tf.reduce_mean(tf.abs(tf.subtract(gram_comp,gram_gt))))
    style_comp_loss = functools.reduce(tf.add,styleloss)
    return style_comp_loss

def Tv_loss(I_comp):
    I_comp_size = [i.value for i in I_comp.get_shape()]
    x_size = int(I_comp_size[2]*(I_comp_size[1]-1)*I_comp_size[3])
    y_size = int(I_comp_size[1]*(I_comp_size[2]-1)*I_comp_size[3])
    I_comp_x1 = I_comp[:,0:(I_comp_size[1]-1),:,:]
    I_comp_x2 = I_comp[:,1:I_comp_size[1],:,:]
    I_comp_y1 = I_comp[:,:,0:(I_comp_size[2]-1),:]
    I_comp_y2 = I_comp[:,:,1:I_comp_size[2],:] 
    tv_x = tf.reduce_sum(tf.squared_difference(I_comp_x1,I_comp_x2))/x_size
    tv_y = tf.reduce_sum(tf.squared_difference(I_comp_y1,I_comp_y2))/y_size
    tv_loss = tv_y+tv_x
    return tv_loss

def get_total_loss(I_out,I_gt,bin_mask,fai_out,fai_gt,fai_comp,layers,I_comp):
    l_hole = hole_loss(I_out,I_gt,bin_mask)
    l_valid = valid_loss(I_out,I_gt,bin_mask)
    percept_loss = Percept_loss(fai_out,fai_gt,fai_comp,layers)
    style_loss_out = Style_loss_out(fai_out,fai_gt,layers)
    style_loss_comp = Style_loss_comp(fai_comp,fai_gt,layers)
    tv_loss = Tv_loss(I_comp)
    all_loss = 6*l_hole + l_valid + 0.05*percept_loss + 120*(style_loss_out + style_loss_comp) + 0.1*tv_loss
    return all_loss,l_hole,l_valid,percept_loss,style_loss_out,style_loss_comp,tv_loss
