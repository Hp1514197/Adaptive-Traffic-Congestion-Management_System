import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("./videos/IndiraNagar.mp4")  #For Video
#cap = cv2.VideoCapture(0)  # For Webcam

model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# mask = cv2.imread('./images/Untitled design.png')

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limit1 = [935, 90, 1275, 90]
limit2 = [935, 110, 1275, 110]

limit3 = [1365, 120, 1365, 360]
limit4 = [1385, 120, 1385, 360]

limit5 = [600, 70, 600, 170]
limit6 = [620, 70, 620, 170]

limit7 = [450, 500, 1240, 500]
limit8 = [450, 520, 1240, 520]



totalCount1 = []
totalCount2 = []
totalCount3 = []
totalCount4 = []

while True:
    success, img = cap.read()
    #success, img2 = cap.read()
    imgRegion = cv2.bitwise_and(img, img)

    # imgGraphics = cv2.imread("./images/graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # for lan 1
    cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (250, 182, 122), 2)
    cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (250, 182, 122), 2)

    # for lan 2
    cv2.line(img, (limit3[0], limit3[1]), (limit3[2], limit3[3]), (250, 182, 122), 2)
    cv2.line(img, (limit4[0], limit4[1]), (limit4[2], limit4[3]), (250, 182, 122), 2)

    # for lan 3
    cv2.line(img, (limit5[0], limit5[1]), (limit5[2], limit5[3]), (250, 182, 122), 2)
    cv2.line(img, (limit6[0], limit6[1]), (limit6[2], limit6[3]), (250, 182, 122), 2)

    # for lan 4
    cv2.line(img, (limit7[0], limit7[1]), (limit7[2], limit7[3]), (250, 182, 122), 2)
    cv2.line(img, (limit8[0], limit8[1]), (limit8[2], limit8[3]), (250, 182, 122), 2)



    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(25, y1)),scale=1, thickness=1,colorR=(56, 245, 213) ,colorT=(25, 26, 25),offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

        #for lan 1
        if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
            if totalCount1.count(id) == 0:
                totalCount1.append(id)
                cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (12, 202, 245), 3)

        if limit2[0] < cx < limit2[2] and limit2[1] - 15 < cy < limit2[1] + 15:
            if totalCount1.count(id) == 0:
                totalCount1.append(id)
                cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (12, 202, 245), 3)
        # for lan 2
        if limit3[1] < cy < limit3[3] and limit3[0] - 15 < cx < limit3[0] + 15:
            if totalCount2.count(id) == 0:
                totalCount2.append(id)
                cv2.line(img, (limit3[0], limit3[1]), (limit3[2], limit3[3]), (0, 255, 0), 5)

        if limit4[1] < cy < limit4[3] and limit4[0] - 15 < cx < limit4[0] + 15:
            if totalCount2.count(id) == 0:
                totalCount2.append(id)
                cv2.line(img, (limit4[0], limit4[1]), (limit4[2], limit4[3]), (0, 255, 0), 5)
        # for lan 3
        if limit5[1] < cy < limit5[3] and limit5[0] - 15 < cx < limit5[0] + 15:
            if totalCount3.count(id) == 0:
                totalCount3.append(id)
                cv2.line(img, (limit5[0], limit5[1]), (limit5[2], limit5[3]), (0, 255, 0), 5)

        if limit6[1] < cy < limit6[3] and limit6[0] - 15 < cx < limit6[0] + 15:
            if totalCount3.count(id) == 0:
                totalCount3.append(id)
                cv2.line(img, (limit6[0], limit6[1]), (limit6[2], limit6[3]), (0, 255, 0), 5)
        # for lan 4
        if limit7[0] < cx < limit7[2] and limit7[1] - 15 < cy < limit7[1] + 15:
            if totalCount4.count(id) == 0:
                totalCount4.append(id)
                cv2.line(img, (limit7[0], limit7[1]), (limit7[2], limit7[3]), (12, 202, 245), 3)

        if limit8[0] < cx < limit8[2] and limit8[1] - 15 < cy < limit8[1] + 15:
            if totalCount4.count(id) == 0:
                totalCount4.append(id)
                cv2.line(img, (limit8[0], limit8[1]), (limit8[2], limit8[3]), (12, 202, 245), 3)

    cvzone.putTextRect(img, f' 1st Lane: {len(totalCount1)}', (25, 75),2,thickness=2,colorR=(147, 245, 186),colorT=(15, 15, 15))
    cvzone.putTextRect(img, f' 2nd Lane: {len(totalCount2)}', (25, 145),2,thickness=2,colorR=(147, 245, 186),colorT=(15, 15, 15))
    cvzone.putTextRect(img, f' 3rd Lane: {len(totalCount3)}', (25, 215),2,thickness=2,colorR=(147, 245, 186),colorT=(15, 15, 15))
    cvzone.putTextRect(img, f' 4th Lane: {len(totalCount4)}', (25, 285),2,thickness=2,colorR=(147, 245, 186),colorT=(15, 15, 15))



    #cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Traffic_Flow_Output",img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)