import os, glob, cv2, shutil

IMAGE_ROOT = '../Images/output/'
LABEL_ROOT = '../Labels/output/'
NEW_IMG_ROOT = './Images/output/'
if not os.path.isdir(IMAGE_ROOT):
    print('No image directory')
    exit
if not os.path.isdir(LABEL_ROOT):
    print('No label directory')
    exit
if not os.path.isdir(LABEL_ROOT):
    print('No new image directory')
    os.mkdir(NEW_IMG_ROOT)

def main():
    imageList = glob.glob(os.path.join(IMAGE_ROOT, '*.jpg'))
    imageList.sort()
    if len(imageList) == 0:
        print('No .jpg images found in the specified dir!')
        return
    labelList = glob.glob(os.path.join(LABEL_ROOT, '*.txt'))
    labelList.sort()
    if len(labelList) == 0:
        print('No .txt labels found in the specified dir!')
        return
    label_idx = 0
    for idx,imagename in enumerate(imageList):
        temp_image_name = imagename.replace(IMAGE_ROOT,'')
        temp_image_name = temp_image_name.replace('.jpg', '')
        temp_lable_name = labelList[label_idx].replace(LABEL_ROOT, '')
        temp_lable_name = temp_lable_name.replace('.txt', '')
        if temp_image_name == temp_lable_name:
            label_idx += 1
            continue
        else:
            new_img_name = NEW_IMG_ROOT + temp_image_name + '.jpg'
            print(imagename, new_img_name)
            shutil.copy(imagename, new_img_name)

if __name__ == "__main__":
    main()