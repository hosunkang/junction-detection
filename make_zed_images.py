import cv2, os
import pyzed.sl as sl
import sys
import argparse

SAVE_ROOT = './data/'
if not os.path.isdir(SAVE_ROOT):
    os.mkdir(SAVE_ROOT)

def main(FLAGS):
    print(FLAGS.num)
    count = FLAGS.num
    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    vid = sl.Camera()
    if not vid.is_opened():
        print("Opening ZED Camera...")
    status = vid.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    while True:
        err = vid.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            vid.retrieve_image(mat, sl.VIEW.VIEW_SIDE_BY_SIDE)
            frame = mat.get_data()
            re_frame = cv2.resize(frame,(1280,360))
            l_frame = frame[0:1080,0:1920]
            r_frame = frame[0:1080,1920:3840]
        else:
            continue

        cv2.imshow('frame',re_frame) 
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            imagename_l = 'l_box_{:04d}.png'.format(count)
            imagename_r = 'r_box_{:04d}.png'.format(count)
            savename_l = SAVE_ROOT + imagename_l
            savename_r = SAVE_ROOT + imagename_r
            cv2.imwrite(savename_l, l_frame)
            cv2.imwrite(savename_r, r_frame)
            count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--num', type=int,
    )

    FLAGS = parser.parse_args()
    main(FLAGS)
