import cv2
import argparse
import numpy as np

class yolov5():
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n') 
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(self.anchors)
        self.na = len(self.anchors[0]) // 2
        self.no = num_classes + 5 
        self.stride = np.array([8., 16., 32.])
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNetFromONNX(yolo_type + '.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
    
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True,scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # if not scaleup:  # only scale down, do not scale up (for better val mAP)
        #     r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # if auto:  # minimum rectangle
        #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
       
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
       # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self,prediction, conf_thres=0.25,agnostic=False):
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
       
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:],axis=1)
            #转为array：  x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            re = np.array(conf.reshape(-1)> conf_thres)
            #转为维度
            conf =conf.reshape(-1,1)
            j = j.reshape(-1,1)
            #numpy的拼接
            x = np.concatenate((box,conf,j),axis=1)[re]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            #转为list 使用opencv自带nms
            boxes = boxes.tolist()
            scores = scores.tolist()
            i = cv2.dnn.NMSBoxes(boxes, scores, self.confThreshold, self.nmsThreshold)
            #i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            output[xi] = x[i]
        return output

    def detect(self, srcimg): 
        im = srcimg.copy()
        im, ratio, wh = self.letterbox(srcimg, self.inpWidth, stride=self.stride, auto=False)
        # Sets the input to the network
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0,(self.inpWidth, self.inpHeight),[0,0,0],swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        #NMS
        pred = self.non_max_suppression(outs, self.confThreshold,agnostic=False)
        #draw box
        for i in pred[0]:
            left = int((i[0] - wh[0])/ratio[0])
            top = int((i[1]-wh[1])/ratio[1])
            width = int((i[2] - wh[0])/ratio[0])
            height = int((i[3]-wh[1])/ratio[1])
            conf = i[4]
            classId = i[5]
            #frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            cv2.rectangle(srcimg, (int(left), int(top)), (int(width),int(height)), (0, 0, 255), thickness=2)
            label = '%.2f' % conf
            label = '%s:%s' % (self.classes[int(classId)], label)
            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            #cv2.rectangle(srcimg, (int(left), int(top - round(1.5 * labelSize[1]))), (int(left + round(1.5 * labelSize[0])), int(top + baseLine)), (255,255,255), cv2.FILLED)
            cv2.putText(srcimg, label, (int(left-20),int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=2)
        cv2.imshow('show', srcimg)
        cv2.waitKey(0)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='data/images/bus.jpg', help="image path")
    parser.add_argument('--net',type=str,default='yolov5s',help='model name')
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    model = yolov5(args.net, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)
    model.detect(srcimg)



