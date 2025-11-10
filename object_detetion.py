import numpy as np
import cv2 as cv

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"LOVE PAI", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "GUITAR","POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

net = cv.dnn.readNetFromCaffe("./Mo/MobileNetSSD.prototxt" , "./Mo/MobileNetSSD.caffemodel")


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        (h,w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
                cv.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv.rectangle(frame, (startX, startY - 15), (endX, startY), COLORS[class_index], cv.FILLED)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv.putText(frame, label, (startX+20, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()