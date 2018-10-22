'''

  Created by irving on 19/10/18

'''
import time
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gpu
import cv2
import numpy as np

threshold = 0.8
# batch_size_list = [1, 1] + list(range(1, 30)) # warmup + testing
batch_size_list = list(range(5, 30))
ctx = mx.gpu(0)
# net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=gpu(0))
# net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True, ctx=gpu(0))
net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)

for batch_size in batch_size_list:
    cap = cv2.VideoCapture('/mnt/sda/videos/office.mp4')
    num_frame = 0
    num_of_batches = 0
    t1 = time.time()
    while cap.isOpened():
        nd_img_list = []
        np_img_list = []

        for i in range(batch_size):
            ret, np_img = cap.read()
            if ret:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, (512, int(512.0/img.shape[0]*img.shape[1])))
                # numpy_image_list.append(img)
                # num_frame += 1
                # frame = mx.nd.array(np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :])
                # image_list.append(frame)
                num_frame += 1
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                nd_img = mx.nd.array(np_img)
                nd_img_list.append(nd_img)
            else:
                break
        if len(nd_img_list) > 0:
            nd_img_list, np_img_list = data.transforms.presets.yolo.transform_test(nd_img_list, short=512)
            num_of_batches += 1
            x = mx.ndarray.concat(*nd_img_list, dim=0).as_in_context(ctx)
            class_IDs, scores, bounding_boxs = net(x)
            # for frame_no in range(len(np_img_list)):
            #     ax = utils.viz.plot_bbox(np_img_list[frame_no], bounding_boxs[frame_no], scores[frame_no],
            #                              class_IDs[frame_no], class_names=net.classes)
            #     plt.pause(0.01)
            # plt.show()
        else:
            break



    t2 = time.time()
    print(f'Processed {num_of_batches} batches of {num_frame} images in {t2-t1}s with batch size of {batch_size}.')
    cap.release()