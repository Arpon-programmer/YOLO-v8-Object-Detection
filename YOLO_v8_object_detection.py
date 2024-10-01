import cv2
from ultralytics import YOLO
import numpy as np
#cap=cv2.VideoCapture('dogs_1.mp4')
cap=cv2.VideoCapture(0)
model=YOLO("yolov8m.pt")
labels=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'pen']

while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame,device='cpu')
    result=results[0]
    bboxes=np.array(result.boxes.xyxy.cpu(),dtype='int')
    classes=np.array(result.boxes.cls.cpu(),dtype='int')
    for cls,bbox in zip(classes,bboxes):
        (x,y,x2,y2)=bbox
        cv2.rectangle(frame,(x,y),(x2,y2),(0,255,255),2)
        cv2.putText(frame, str(labels[cls]), (x, y - 5),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv2.imshow('Object-Detection', frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
