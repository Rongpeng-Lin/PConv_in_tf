# PConv_in_tf
&#8195;'Image Inpainting for Irregular Holes Using Partial Convolutions' first tensorflow primary instance, fully implemented using tensorflow, without modifying the source code.<br>
## Partial Conv
&#8195;Use curr_bin_mask to represent the mask of the current binary; conved_mask represents the result of convolution of the binary mask, corresponding to sum(M) in the text; new_bin_mask represents the new binary mask after convolution, and the update rule is:<br>&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;((conved_mask==0)==0)<br>therefore, the local convolution is calculated as follows:<br>&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;Pconv(x) = (Conv(x*curr_bin_mask)*conved_mask+b)*new_bin_mask<br>The operation with the new mask is to ensure that the invalid input is zero, as described in the text.
## Network structure
&#8195;[U_net structure diagram:](https://arxiv.org/abs/1411.4038)<br>&#8195;![image](https://github.com/Rongpeng-Lin/PConv_in_tf/blob/master/U_net/u_net_Struct.png)<br>Replace convolution with local convolution<br>
## Mask generation
&#8195;Unlike the original,I used opencv to generate a mask and set the invalid part input to zero. In order to ensure the irregularity of the mask, without filling, the number of units in the mask part is also random, but at least a total of 4 * 5 units, up to 12 * 5 units (these can be set).<br>
## Use
### 1.Generate images and masks:
&#8195;&#8195;&#8195;&#8195;num_mask:  the number of generated masks<br>&#8195;&#8195;&#8195;&#8195;min_units:  the lower limit of the number of occlusion units in the mask<br>&#8195;&#8195;&#8195;&#8195;max_units:  the upper limit of the occlusion unit in the mask<br>&#8195;&#8195;&#8195;&#8195;new_mask_path:  the storage path of the generated mask<br>&#8195;&#8195;&#8195;&#8195;im_file:  the original image path<br>&#8195;&#8195;&#8195;&#8195;new_im_path:  The mask acts on the path after the original image,this is the training sample we generated.<br>Example of use:<br>&#8195;&#8195;&#8195;python config_im_mask.py --num_mask=6 --min_units=5 --max_units=12 --new_mask_path="D:/inpaint/masks/" --im_file="D:/inpaint/imfiles/" --new_im_path="D:/inpaint/imfilenew/"<br>
### 2.Training:
