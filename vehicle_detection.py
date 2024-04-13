import cv2

# Load YOLOv4
net = cv2.dnn.readNet("yolov8l.weights", "yolov8l.cfg")
# If you're using YOLOv5, change the filenames accordingly (e.g., yolov5.weights, yolov5.cfg)

# Load input image
image = cv2.imread("input_image.jpg")
height, width, _ = image.shape

# Preprocess input image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Forward pass through the network
outs = net.forward(output_layers)

# Process detection results
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = cv2.dnn.NMSBoxes(detection[:4], scores, score_threshold=0.5, nms_threshold=0.4)
        if len(class_id) > 0:
            for i in class_id.flatten():
                box = detection[:4]
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Vehicle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
cv2.imwrite("output_images", image)

# Show the output image
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
