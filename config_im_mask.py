from create_mask_im import *
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_mask', type=int,
                        dest='num_mask', help='how many mask to generate',
                        metavar='Number of mask', required=True)
    
    parser.add_argument('--min_units', type=int,
                        dest='min_units', help='min units to generate',
                        metavar='Min units to generate', required=True)
    
    parser.add_argument('--max_units', type=int,
                        dest='max_units', help='max units to generate',
                        metavar='Max units to generate', required=True)

    parser.add_argument('--new_mask_path', type=str,
                        dest='new_mask_path', help='path to save masks',
                        metavar='Path to save masks', required=True)

    parser.add_argument('--im_file', type=str,
                        dest='im_file', help='path to raw image',
                        metavar='Path to raw image', required=True)

    parser.add_argument('--new_im_path', type=str,
                        dest='new_im_path', help='image to train',
                        metavar='Train', required=True)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    save_mask(options.num_mask, options.min_units, options.max_units, options.new_mask_path, options.im_file, options.new_im_path)
    
if __name__ == '__main__':
    main()
