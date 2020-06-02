# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import cv2
import pyzed.sl as sl
import copy as cp
import math
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


class YOLO(object):
    _defaults = {
        "model_path": 'logs/4000.h5',
        "anchors_path": 'model_data/my_anchors.txt',
        "classes_path": 'model_data/my_classes.txt',
        "score" : 0.3,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.baseline = 120 #ZED mini = 63, zed = 120
        self.focal = 1400

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, image_right):
        start = timer()
        if not image_right == None:
            r_in_junction, r_out_junctions = self.detect_image_right(image_right)
        width , height = image.size
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(5e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        out_junctions = []
        in_junction = []
        angles = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box  # Information of each boxes
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if predicted_class == "in_junc":
                in_center_x = (left + right ) // 2
                in_center_y = (top + bottom ) // 2
                in_junction.append(in_center_x)
                in_junction.append(in_center_y)
                area = (left, top, right, bottom)
                ROI = image.crop(area)
                angles = self.detect_line(ROI)
                # print(int(math.degrees(angles[0])), int(math.degrees(angles[1])), int(math.degrees(angles[2])))
                

            elif predicted_class == "out_junc":
                center_x = (left + right ) // 2
                center_y = (top + bottom ) // 2
                out_junctions.append([center_x,center_y])
            # else: ## box
            #     area = (left, top, right, bottom)
            #     ROI = image.crop(area)
            #     self.test(ROI)
            #     # for i in range(thickness):
            #     #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])

        ##### Draw the box size line
        draw = ImageDraw.Draw(image)
        good_out_junc = []
        if in_junction != [] and out_junctions != []:
            r = 20
            draw.ellipse((in_junction[0]-r, in_junction[1]-r, in_junction[0]+r, in_junction[1]+r),(0,255,0), (0,255,0))
            if r_in_junction != []:
                in_disparity = in_junction[0] - r_in_junction[0]
                in_x = self.baseline * (in_junction[0]-width//2) / in_disparity
                in_y = self.baseline * (in_junction[1]-height//2) / in_disparity
                in_z = self.baseline * self.focal / in_disparity
                p1 = [in_x,in_y,in_z]

            if len(out_junctions) == 6:
                for angle in angles:
                    min_angle_gap = 360
                    min_angle = [0,0]
                    for out_junction in out_junctions: 
                        temp_angle = self.get_angle(in_junction[0], in_junction[1], out_junction[0], out_junction[1]) + 180
                        if abs(temp_angle - math.degrees(angle)) < min_angle_gap:
                            min_angle_gap = abs(temp_angle - math.degrees(angle))
                            min_angle = out_junction      
                    draw.line((in_junction[0], in_junction[1], min_angle[0], min_angle[1]), (0,255,0), 10)
                
                    if r_out_junctions != []:
                        min_dis = 100000
                        min_r_junc = [0,0]
                        for r_out_junction in r_out_junctions:
                            if abs(r_out_junction[1] - min_angle[1]) < 50:
                                temp_dis = self.euclidean_distance2(min_angle, r_out_junction)
                                if temp_dis < min_dis:
                                    min_dis = temp_dis
                                    min_r_junc = r_out_junction
                            
                        disparity = min_angle[0] - min_r_junc[0]
                        out_x = self.baseline * (min_angle[0]-width//2) / disparity
                        out_y = self.baseline * (min_angle[1]-height//2) / disparity
                        out_z = self.baseline * self.focal / disparity
                        p2 = [out_x,out_y,out_z]

                        if p1 != [] and p2 != []:
                            distance = self.euclidean_distance3(p1,p2)
                            
                            text_distance = "%.2f mm"%(distance)
                            draw.text((min_angle[0]+10, min_angle[1]-10), text_distance, fill=(0, 0, 255), font=font)
                            draw.ellipse((min_angle[0]-r, min_angle[1]-r, min_angle[0]+r, min_angle[1]+r),(255,0,0), (255,0,0))
        del draw
        end = timer()
        return image

    def detect_image_right(self, image):
        width , height = image.size

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        temp_in_junction = []
        temp_out_junctions = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            draw = ImageDraw.Draw(image)

            top, left, bottom, right = box  # Information of each boxes
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
            if predicted_class == "in_junc":
                in_center_x = (left + right ) // 2
                in_center_y = (top + bottom ) // 2
                temp_in_junction.append(in_center_x)
                temp_in_junction.append(in_center_y)

            elif predicted_class == "out_junc":
                center_x = (left + right ) // 2
                center_y = (top + bottom ) // 2
                temp_out_junctions.append([center_x, center_y])

        return temp_in_junction, temp_out_junctions

    def detect_line(self, img_ori):
        img = np.asarray(img_ori)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 10,40, L2gradient = True)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        result = cv2.dilate(edge, k)
        # cv2.imshow('ddd', result)
        h, w  = result.shape[:2]
        result2 = cv2.morphologyEx(result, cv2.MORPH_OPEN, k)
        result_for_cen = cv2.erode(result2,k)
        h_for_cen, w_for_cen = result_for_cen.shape[:2]
        mindis = h_for_cen+w_for_cen
        temp = 0
        cen_x = w_for_cen // 2
        cen_y = h_for_cen // 2
        for i in range(h_for_cen):
            for j in range(w_for_cen):
                if result_for_cen[i,j] == 255:
                    temp = self.get_distance(w_for_cen//2, h_for_cen//2, j, i)
                    if temp < mindis:
                        mindis = temp
                        cen_x, cen_y = j, i
        angles = []
        for i in range(h):
            for j in range(w):
                if result[i,j] == 255:
                    angles.append(self.get_angle(cen_x, cen_y, j, i))
        
        degree_hist = []
        for i in range(-179,181,1):
            temp_hist = 0
            for angle in angles:
                if angle == i:
                    temp_hist += 1
            degree_hist.append(temp_hist)
        

        de0_119 = degree_hist[0:119]
        de120_239 = degree_hist[120:239]
        de240_359 = degree_hist[240:359]
        degree1 = math.radians(np.argmax(de0_119))
        degree2 = math.radians(np.argmax(de120_239)+120)
        degree3 = math.radians(np.argmax(de240_359)+240)
        degrees = []
        degrees.append(degree1)
        degrees.append(degree2)
        degrees.append(degree3)
        
        # cv2.line(img, (cen_x, cen_y), (int(cen_x - 50*math.cos(degree1)), int(cen_y - 50*math.sin(degree1))),(0,255,0), 3)
        # cv2.line(img, (cen_x, cen_y), (int(cen_x - 50*math.cos(degree2)), int(cen_y - 50*math.sin(degree2))),(0,255,0), 3)
        # cv2.line(img, (cen_x, cen_y), (int(cen_x - 50*math.cos(degree3)), int(cen_y - 50*math.sin(degree3))),(0,255,0), 3)
        #cv2.imshow('center_point', img)

        return degrees
    
    def close_session(self):
        self.sess.close()
    
    def euclidean_distance2(self, p1, p2):
        temp_dis = math.sqrt(pow(abs(p1[0]-p2[0]),2) + pow(abs(p1[1]-p2[1]),2))
        return temp_dis

    def euclidean_distance3(self, p1, p2):
        temp_dis = math.sqrt(pow(abs(p1[0]-p2[0]),2) + pow(abs(p1[1]-p2[1]),2)+ pow(abs(p1[2]-p2[2]),2))
        return temp_dis

    def get_distance(self, w_c, h_c, w ,h):
        temp_dis = math.sqrt(pow(abs(w_c-w),2) + pow(abs(h_c-h),2))
        return temp_dis
    
    def get_angle(self, w_c, h_c, w ,h):
        temp_angle = int(math.degrees(math.atan2(h-h_c, w-w_c)))
        return temp_angle

def detect_video(yolo, video_path, output_path=""):
    if video_path == "webcam": #real time ZED camera 
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
        mat_r = sl.Mat()
        isOutput = True if output_path != "" else False
        if isOutput:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(output_path, fourcc, 25.0, (640,480))
    else:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        if video_path == "webcam":
            err = vid.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                vid.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
                frame = mat.get_data()
                image = Image.fromarray(frame)

                vid.retrieve_image(mat_r, sl.VIEW.VIEW_RIGHT)
                frame_r = mat_r.get_data()
                image_r = Image.fromarray(frame_r)
            else:
                continue
        else:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
        image= yolo.detect_image(image,image_r)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

