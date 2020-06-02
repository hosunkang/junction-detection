# Junction detection Demo code
## REFERENCE
1. <https://github.com/qqwweee/keras-yolo3>
2. <https://github.com/hosunkang/segment_annotation_tool>
3. <https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New->

---
## REQUIREMENTS
pyzed, keras, opencv, matplotlib, pillow, numpy, os, tensorflow

## USAGE
1. Get your custom box images
2. Make annotation files : `python annotation_tool.py`
  - Input images folder name (ex.'output/', 'box/') --> Root) Images/output/...
3. Make train.txt file : `python make_train.txt.py`
4. Train : `python train.py`
5. Evaluation : `python yolo.py` or `python yolo_video.py`
