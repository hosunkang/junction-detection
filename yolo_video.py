import sys
import argparse, cv2, glob, os
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    count = 1
    img_l_dir = 'testimage/left/'
    img_r_dir = 'testimage/right/'
    img_ls = glob.glob(os.path.join(img_l_dir, '*.png'))
    img_ls.sort()
    img_rs = glob.glob(os.path.join(img_r_dir, '*.png'))
    img_rs.sort()
    
    for idx, img_l in enumerate(img_ls):
        if img_rs == []:
            try:
                image = Image.open(img_l)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, None)
                r_image_np = np.asarray(r_image)
                r_image_np = cv2.cvtColor(r_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite('result_{0}.png'.format(count), r_image_np)
                count += 1
                #r_image.show()    
        else:
            try:
                image = Image.open(img_l)
                image_right = Image.open(img_rs[idx])
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image,image_right)
                r_image_np = np.asarray(r_image)
                r_image_np = cv2.cvtColor(r_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite('result_{0}.png'.format(count), r_image_np)
                count += 1
                #r_image.show()   

    # while True:
    #     img_l = input('Input left image filename:')
    #     if img_l == 'q':
    #         break

    #     img_r = input('Input right image filename:')
    #     if img_r == None:
    #         try:
    #             image = Image.open(img_l)
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             r_image = yolo.detect_image(image, None)
    #             r_image_np = np.asarray(r_image)
    #             r_image_np = cv2.cvtColor(r_image_np, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite('result_{0}.png'.format(count), r_image_np)
    #             count += 1
    #             #r_image.show()    
    #     else:
    #         try:
    #             image = Image.open(img_l)
    #             image_right = Image.open(img_r)
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             r_image = yolo.detect_image(image,image_right)
    #             r_image_np = np.asarray(r_image)
    #             r_image_np = cv2.cvtColor(r_image_np, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite('result_{0}.png'.format(count), r_image_np)
    #             count += 1
    #             #r_image.show()   
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
