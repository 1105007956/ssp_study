import numpy as np
import cv2
import os


def convert(img,lines):
    data_box = []
    data_box.append(img)
    img = os.path.join('C:\\Users\\ssp\\Desktop\\keras-yolo3-master' , img)
    img = cv2.imread(img)
    h,w,_ = img.shape


    for line in lines:
        box = line.split()
        class_name = box[0]
        box[1] = float(box[1]) * w#长中心
        box[2] = float(box[2]) * h#宽中心
        box[3] = (float(box[3]) * w)/2#框长
        box[4] = (float(box[4]) * h)/2#框宽
        xmin = box[1] - box[3]#ymin
        ymin = box[2] - box[4]#xmin
        xmax = box[1] + box[3]  # ymax
        ymax= box[2] + box[4]  # xmax
        box[0] = round(xmin)
        box[1] = round(ymin)
        box[2] = round(xmax)
        box[3] = round(ymax)
        box[4] = 0
        box_str = ''
        i = 0
        for x in box:
            i=i+1
            box_str = box_str+str(x)
            if i!=5:
                box_str = box_str + ','
        data_box.append(box_str)

    return data_box



labels_path = 'C:\\Users\\ssp\\Desktop\\keras-yolo3-master\\training_data\\label'
img_path = 'training_data/cable/'
labels = os.listdir(labels_path)
# training_validation_split = 0.9
# batch_size = 20
# data_img = []
# data_label = []
sspall= []
yy = 0

for label in labels:
    base_name = label.split('.')[0]
    image_path = os.path.join(img_path, base_name + '.jpg')
    label_path = os.path.join(labels_path, label)
    f = open(label_path)
    lines = f.readlines()
    print(yy)
    data_change = convert(image_path,lines)
    ssp = ''
    j = 0
    for d_c in data_change:
        ssp =ssp + d_c
        j = j + 1
        if j!=len(data_change):
            ssp = ssp + ' '
    sspall.append(ssp)
all='\n'.join(str(xx) for xx in sspall)
file = open('file_name.txt', 'w')
file.write(all)
file.close()
print('ss')
