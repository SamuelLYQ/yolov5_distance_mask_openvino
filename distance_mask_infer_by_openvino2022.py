# Do the inference by OpenVINO2022.1

import argparse
import cv2
import numpy as np
import time
from numpy import random
import torch
import yaml
from openvino.runtime import Core  # the version of openvino >= 2022.1
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox

# 文件路径
vid_path = "./videos/video.mp4"
# vid_path = "./videos/face_mask.jpg"

# 载入COCO Label
with open('./coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


# 目标检测函数，返回检测结果
def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    preds = net([blob])[next(iter(net.outputs))] # API version>=2022.1
    return preds

# YOLOv5的后处理函数，解析模型的输出
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    #print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        if class_list[class_ids[i]] == "person":
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

# 按照YOLOv5 letterbox resize的要求，先将图像长:宽 = 1:1，多余部分填充黑边
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0

def detectDistanceAndMask():
    imgsz = 640
    vid_path = opt.source
    DEVICE = opt.device
    # 载入yolov5s xml or onnx模型
    model_path = "yolov5s.xml"
    ie = Core() #Initialize Core version>=2022.1
    net = ie.compile_model(model=model_path, device_name="AUTO")

    # 载入mask模型
    mask_model_path = "best.xml"
    mask_net = ie.compile_model(model=mask_model_path, device_name="AUTO")
    mask_model = attempt_load('best.pt', map_location=DEVICE)
    imgsz = check_img_size(imgsz, s=mask_model.stride.max())
    # Get names and colors
    names = ['unmasked', 'masking']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 开启Webcam，并设置为1280x720
    #cap = cv2.VideoCapture(0)

    # 读取mp4
    cap = cv2.VideoCapture(vid_path)
    writer = None
    (W, H) = (None, None)

    half = False

    img = torch.zeros((1, 3, imgsz, imgsz), device=DEVICE)  # init img

    # 开启检测循环
    while True:
        start = time.time()
        _, frame = cap.read()
        if frame is None:
            print("End of stream")
            break

        # 将图像按最大边1:1放缩
        inputImage = format_yolov5(frame)
        # 执行推理计算
        outs = detect(inputImage, net)
        # preds = detect(inputImage, mask_net)
        #mask handle
        img = letterbox(frame, new_shape=imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(DEVICE)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = mask_model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, 2, False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            #     p, s, im0 = path, '', im0s
            p, s, im0 = vid_path, '', frame
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # cv2.imshow(p, im0)
            # if cv2.waitKey(1) == ord('q'):  # q to quit
            #     raise StopIteration

        # if W is None or H is None:
        #     (H, W) = frame.shape[:2]
        #     q = W
        #
        # frame = frame[0:H, 200:q]
        (H, W) = frame.shape[:2]

        # 拆解推理结果
        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        status = list()
        close_pair = list()
        s_close_pair = list()
        center = list()
        dist = list()
        # 显示检测框bbox
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(0)

        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(center[i], center[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):

            black_rect = np.ones(inputImage.shape, dtype=np.uint8) * 0

            res = cv2.addWeighted(inputImage, 0.77, black_rect, 0.23, 1.0)

            cv2.putText(frame, "Distance & Mask Analyser wrt. COVID-19", (210, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (20, 60), (510, 160), (170, 170, 170), 2)
            cv2.putText(frame, "Connecting lines shows closeness among people. ", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "-- YELLOW: CLOSE", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "--    RED: VERY CLOSE", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.rectangle(frame, (535, 60), (W - 20, 160), (170, 170, 170), 2)
            cv2.putText(frame, "Bounding box shows the level of risk to the person.", (545, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "-- DARK RED: HIGH RISK", (565, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
            cv2.putText(frame, "--   ORANGE: LOW RISK", (565, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

            cv2.putText(frame, "--    GREEN: SAFE", (565, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            tot_str = "TOTAL COUNT: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "LOW RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)

            black_rect = np.ones(inputImage.shape, dtype=np.uint8) * 0

            res = cv2.addWeighted(inputImage, 0.8, black_rect, 0.2, 1.0)

            cv2.putText(frame, tot_str, (10, H - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, safe_str, (10, H - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, low_str, (10, H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 1)
            cv2.putText(frame, high_str, (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)

            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

        # 显示推理速度FPS
        end = time.time()
        inf_end = end - start
        fps = 1 / inf_end
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(fps_label+ "; Detections: " + str(len(class_ids)))
        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='videos/video.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='cpu', help='GPU or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
         detectDistanceAndMask()