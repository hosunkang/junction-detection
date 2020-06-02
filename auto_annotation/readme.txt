For make annotation for YOLOv3

USAGE 1.
python annotation_tool.py

Execute annotation_tool and then input the image direction. In my case, input the 'output/'.

USAGE 2.
This USAGE is for someone who have custom model file(logs/000/trained_weight_final.h5)

python auto_labeling.py --image

For this USAGE, you need to make sure that target image are in the Images/ouput folder.


#######################3
annotation_tool.py => Customize own dataset tool. (self work)
auto_labeling.py   => Customize own dataset tool. (auto work). but you need small size own model.
gather_imgs.py     => Gather images have no label to the ./Images/output folder
labeling_yolo.py   => YOLOv3 python file for auto_labeling.py (return the label => x1,y1,x2,y2,class)
