from main import *
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epoch', type=int,
                        dest='num_epoch', help='how many epochs to train',
                        metavar='Number of epochs', required=True)

    parser.add_argument('--save_path', type=str,
                        dest='save_path', help='path to save ckpt',
                        metavar='Path to save ckpt', required=True)

    parser.add_argument('--logdir', type=str,
                        dest='logdir', help='path to save net graph',
                        metavar='Path to save net graph', required=True)

    parser.add_argument('--im_path', type=str,
                        dest='im_path', help='path to load images with masks',
                        metavar='Path to load images with masks', required=True)

    parser.add_argument('--vgg_path', type=str,
                        dest='vgg_path', help='where you download vgg.mat',
                        metavar='Where you download vgg.mat', required=True)
    
    
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    image_inpainting = Image_inpainting(options.im_path, options.vgg_path, options.num_epoch, options.logdir, options.save_path)
    image_inpainting.train()
    
if __name__ == '__main__':
    main()
