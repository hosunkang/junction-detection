import argparse
import os, sys
from labeling_yolo import YOLO
from PIL import Image
import os, glob, cv2
from operator import itemgetter

IMAGE_ROOT = './Images/output/'
LABEL_ROOT = './Labels/output/'
if not os.path.isdir(LABEL_ROOT):
    os.mkdir(LABEL_ROOT)


def detect_img(yolo):
    imageList = glob.glob(os.path.join(IMAGE_ROOT, '*.jpg'))
    imageList.sort()
    if len(imageList) == 0:
            print('No .jpg images found in the specified dir!')
            return
    for imagename in imageList:
        labelname = imagename.replace(IMAGE_ROOT, LABEL_ROOT)
        labelname = labelname.replace('.jpg', '.txt')
        in_junc = []
        out_junc = []
        box = []
        image = Image.open(imagename)
        labels = yolo.detect_image(image)
        labels.sort(key=itemgetter(4))
        with open(labelname, 'w') as f:
            for label in labels:
                for idx, i in enumerate(label):
                    label[idx] = str(i)
                label_str = imagename + ' ' + ','.join(label) + '\n'
                #print(label_str)
                f.write(label_str)

FLAGS = None

if __name__ == "__main__":
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