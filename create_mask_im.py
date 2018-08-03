import scipy.misc as misc
import numpy as np
import cv2
import os

def draw_line(imraw,num_line):
    points = np.random.randint(0,511,[num_line,4])
    thick = np.random.randint(10,30,num_line)
    for i in range(num_line):
        p1_start = points[i,:][0:2]
        p1_end = points[i,:][2:]
        cv2.line(imraw,tuple(p1_start),tuple(p1_end),0,thick[i])
    return imraw

def draw_rect(imraw,num_rect,size_max):
    p_starts = np.random.randint(0,512-size_max,[num_rect,2])
    thick = np.random.randint(10,30,num_rect)
    for i in range(num_rect):
        p_start = p_starts[i,:]
        p_end = (p_start[0]+np.random.randint(5,size_max,1)[0],p_start[1]+np.random.randint(5,size_max,1)[0])   
        cv2.rectangle(imraw,tuple(p_start),tuple(p_end),0,thick[i])
    return imraw

def draw_oval(imraw,num_oval):
    centers = np.random.randint(0,512,[num_oval,2])
    lens = np.random.randint(50,300,[num_oval,2])
    total_rots = np.random.randint(-181,181,[num_oval,3])
    thick = np.random.randint(10,30,num_oval)
    for i in range(num_oval):
        cv2.ellipse(imraw,tuple(centers[i]),tuple(lens[i]),total_rots[i][0],total_rots[i][1],total_rots[i][2],0,thick[i])            
    return imraw

def draw_circle(imraw,num_cir):
    centers = np.random.randint(0,512,[num_cir,2])
    Rs = np.random.randint(0,30,num_cir)
    thick = np.random.randint(10,30,num_cir)
    for i in range(num_cir):
        cv2.circle(imraw,tuple(centers[i]),Rs[i],0,thick[i])
    return imraw

def merge(im,mask):
    zeros = np.zeros([512,1024,3])
    zeros[:,:512,:] = im
    im_copy = np.copy(im)
    for i in range(3):
        im_copy[:,:,i] = im_copy[:,:,i]*mask
    zeros[:,512:,:] = im_copy
    return zeros

def save_mask(num_mask, min_units, max_units, new_mask_path, im_file, new_im_path):
    mask_list = []
    for j in range(num_mask):
        mask_init = np.ones([512,512],np.uint8)

        lines = np.random.randint(min_units,max_units,1)[0]
        cicles = np.random.randint(min_units,max_units,1)[0]
        rects = np.random.randint(min_units,max_units,1)[0]
        ovals = np.random.randint(min_units,max_units,1)[0]

        mask_L = draw_line(mask_init,lines)             ##  draw  line
        mask_LR = draw_rect(mask_L,rects,40)            ##  draw  line and rect
        mask_LRO = draw_oval(mask_LR,ovals)        
        mask_list.append(draw_circle(mask_LRO,cicles))
                   ###   save masks
    for j in range(num_mask):
        mask_to_save = np.ones([512,512,3]).astype(np.float32)
        for i in range(3):
            mask_to_save[:,:,i] = mask_to_save[:,:,i] * mask_list[j]
        name = new_mask_path+'mask'+str(j)+'.jpg'
        misc.imsave(name,mask_to_save)
                   ###   save im_with_mask    
    im_dirs = [im_file+i_n for i_n in os.listdir(im_file)]
    for im_dir in im_dirs:
        imraw = misc.imread(im_dir)
        im = misc.imresize(imraw,[512,512])
        for i in range(num_mask):
            im_mask = merge(im,mask_list[i])
            im_mask_name = new_im_path + (im_dir.split('.')[0]).split('/')[-1] +str(i) + '.jpg'       
            misc.imsave(im_mask_name,im_mask)
            
# save_mask(num_mask, min_units, max_units, new_mask_path, im_file, new_im_path)
# save_mask(6, 5, 12, 'D:/nvidainpaint/我写的mask生成程序结果/masks/', 'D:/nvidainpaint/我写的mask生成程序结果/imfiles/', 'D:/nvidainpaint/我写的mask生成程序结果/imfilenew/') 
