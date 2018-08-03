 #######   get_input_im
# def split(im):
#     im1 = im[:,0:512,:]
#     im2 = im[:,512:,:]
#     return im1,im2

# def get_im(aim):
#     im_float = tf.image.convert_image_dtype(tf.image.decode_jpeg(aim, channels=3),tf.float32)
#     im_big = tf.image.resize_images(im_float, [512,1024])
#     im_1,im_2 = split(im_big)
#     im_1_pre,im_2_pre = im_1*255,im_2*255       
#     return im_1_pre,im_2_pre
import scipy.misc as misc

def get_feedict(values,keys):
    dic = {}
    for i in range(len(values)):
        dic[keys[i]] = values[i]
    return dic

def get_im(im_dir,num):
    im_array = misc.imread(im_dir[num])
    im_m,im_g = im_array[:,0:512,:],im_array[:,512:,:]
    return im_m[None,:,:,:],im_g[None,:,:,:]
    
