'''

  Created by irving on 19/10/18

'''
import time
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=6)
args = parser.parse_args()

threshold = 0.8
# cls_of_interest = [0, 2, 5, 7]
cls_of_interest = [5, 6, 14]

ctx = mx.gpu(0)
# net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=ctx)
# net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True, ctx=ctx)
# net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)


net.hybridize(True)
batch_size = args.batch_size
display = True

cap = cv2.VideoCapture('/mnt/sda/videos/IGA1_BREAD_IGA1_20180522162352_20180522165936_306059.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
v_width = cap.get(3)
v_height = cap.get(4)

batch_list = [6] + list(range(1, 30))

for batch_size in batch_list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    num_frame = 0
    num_of_batches = 0
    t1 = time.time()
    while cap.isOpened():
        nd_img_list = []
        np_img_list = []

        nd_img = None
        np_img = None

        for i in range(batch_size):
            ret, np_img = cap.read()
            if ret:
                num_frame += 1
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                nd_img = mx.nd.array(np_img)
                nd_img_list.append(nd_img)
            else:
                break
        if len(nd_img_list) > 0:
            # we need to make the size of each input tensor the same
            # otherwise mxnet will crash
            rest_cnt = batch_size - len(nd_img_list)
            for cnt in range(rest_cnt):
                nd_img_list.append(nd_img_list[-1])
            nd_img_list, np_img_list = data.transforms.presets.yolo.transform_test(nd_img_list, short=512)
            # gluon return the content instead of list
            # here we do a reverse
            # in the future we can change the source file of gluon
            if len(nd_img_list) == 1:
                nd_img_list = [nd_img_list]
                np_img_list = [np_img_list]
            num_of_batches += 1
            x = mx.ndarray.concat(*nd_img_list, dim=0).as_in_context(ctx)
            # print(x.shape, num_frame)
            class_IDs, scores, bounding_boxs = net(x)

            if display:

                class_IDs, scores, bounding_boxs = \
                    class_IDs.asnumpy(), scores.asnumpy(), bounding_boxs.asnumpy()

                for frame in range(batch_size):
                    score = scores[frame, :, :].squeeze()
                    class_id = class_IDs[frame, :, :].squeeze()
                    bounding_box = bounding_boxs[frame, :, :].squeeze().astype(int)

                    index = score > threshold

                    class_id = class_id[index]
                    bounding_box = bounding_box[index, :]

                    show_img = cv2.cvtColor(np_img_list[frame], cv2.COLOR_RGB2BGR)

                    for obj_no in range(bounding_box.shape[0]):
                        if class_id[obj_no] not in cls_of_interest:
                            continue
                        x1, y1, x2, y2 = bounding_box[obj_no, :]
                        cv2.rectangle(show_img, (x1, y1), (x2, y2), (255, 0, 0))

                    cv2.imshow('Test', show_img)
                    cv2.waitKey(1)
        else:
            break

        # del nd_img_list, nd_img, np_img_list, np_img, class_IDs, scores, bounding_boxs, x

    t2 = time.time()
    print(f'video length: {num_frame/fps}s, size: {v_width}*{v_height}, fps: {fps}, processed in {t2-t1}s with batch size of {batch_size}.')
    # cap.release()
