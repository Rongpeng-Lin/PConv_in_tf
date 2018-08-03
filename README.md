# PConv_in_tf
&#8195;'Image Inpainting for Irregular Holes Using Partial Convolutions' first tensorflow primary instance, fully implemented using tensorflow, without modifying the source code.<br>
## Partial Conv
&#8195;Use curr_bin_mask to represent the mask of the current binary; conved_mask represents the result of convolution of the binary mask, corresponding to sum(M) in the text; new_bin_mask represents the new binary mask after convolution, and the update rule is:<br>&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;((conved_mask==0)==0)<br>therefore, the local convolution is calculated as follows:<br>&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;Pconv(x) = (Conv(x*curr_bin_mask)*conved_mask+b)*new_bin_mask<br>The operation with the new mask is to ensure that the invalid input is zero, as described in the text.
## Network structure
&#8195;[U_net structure diagram](https://arxiv.org/abs/1411.4038)
