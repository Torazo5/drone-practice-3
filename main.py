from djitellopy import tello
import cv2
import math
from ultralytics import YOLO  # Make sure to import YOLO from Ultralytics
import time

# Set the desired time interval (in seconds)
interval = 0.5

# Get the current time at the start of the loop
start_time = time.time()

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

# Initialize Tello drone
drone = tello.Tello()
drone.connect()
drone.streamon()
drone.takeoff()
time.sleep(1)
current_height = drone.get_height()
drone.move_down(40)
print(f'BATTERY: {drone.get_battery}')
# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")
done = False
count = 0
rotated = 0
while True:
    if (current_height >= 150):
        break
    if rotated == 24:
        #turned a full cycle
        rotated = 0
        drone.move_up(20)
    elapsed_time = time.time() - start_time
    if elapsed_time >= interval:
        # Reset the start time for the next interval
        start_time = time.time()

        # Your code to run every half-second
        drone.rotate_clockwise(15)
        rotated += 1
    # Get frame from Tello camera
    frame = drone.get_frame_read().frame
    # Perform object detection with YOLO model
    results = model(frame, stream=True)

    #movement
    current_height = drone.get_height()
    print(f'CURRENT HEIGHT: {current_height}')
    #height is only affected my movement from the drone, hand movement doesn't change anything

    # Rest of the code remains the same
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if classNames[int(box.cls[0])] == 'cell phone':
                box_color = (0, 255, 0)  # Green color for cell phone
            else:
                box_color = (255, 0, 255)  # Default color

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            #print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            if classNames[cls] == 'cell phone':
                cv2.putText(frame, classNames[cls], org, font, fontScale, (0,255,0), thickness)
                cv2.imwrite('photo_of_phone/phone_found.jpg', frame)
                done = True
                break
            else:
                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Tello Camera', frame)
    if cv2.waitKey(1) == ord('q') or done == True:
        break

# Land the Tello drone
drone.land()
cv2.destroyAllWindows()
