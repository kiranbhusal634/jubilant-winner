import cv2
import torch
import numpy as np
import serial
import time

# Initialize serial communication with Arduino
arduino_port = "COM12"  # Replace with your Arduino's port
baud_rate = 9600
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to establish

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.half()  # Enable half-precision (if supported)

# Define rotated ROI parameters
ROI_CENTER = (350, 500)
ROI_SIZE = (350, 600)
ROI_ANGLE = -35

def get_rotated_roi(center, size, angle):
    w, h = size
    rect = np.float32([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated_rect = cv2.transform(np.array([rect]), rotation_matrix)
    rotated_rect += np.array(center)
    return np.int32(rotated_rect[0])

ROTATED_ROI = get_rotated_roi(ROI_CENTER, ROI_SIZE, ROI_ANGLE)

vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

# Function to send LED control commands
def control_led(command):
    arduino.write(f"{command}\n".encode('utf-8'))
    print(f"Sent command: {command}")

# Function to process each frame
def process_frame(frame):
    frame = cv2.resize(frame, (1280, 720))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    detections = results.pandas().xyxy[0]
    vehicles = detections[detections['name'].isin(vehicle_classes)]

    count = 0
    for _, row in vehicles.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cv2.pointPolygonTest(ROTATED_ROI, (cx, cy), False) >= 0:
            count += 1
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.polylines(frame, [ROTATED_ROI], isClosed=True, color=(0, 0, 255), thickness=2)
    return frame, count

# Open video
cap = cv2.VideoCapture('highway.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
previous_state = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    processed_frame, vehicle_count = process_frame(frame)

    # Determine LED state based on vehicle count
    if vehicle_count == 3 and previous_state != "green":
        control_led("led2 on")  # Yellow light on
        time.sleep(1)           # Wait for 1 second
        control_led("led2 off") # Yellow light off
        control_led("led1 on")  # Green light on
        control_led("led3 off") # Red light off
        previous_state = "green"
    elif vehicle_count < 3 and previous_state != "red":
        control_led("led2 on")  # Yellow light on
        time.sleep(1)           # Wait for 1 second
        control_led("led2 off") # Yellow light off
        control_led("led1 off") # Green light off
        control_led("led3 on")  # Red light on
        previous_state = "red"

    # Display vehicle count on the frame
    cv2.putText(processed_frame, f"Vehicle Count: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Vehicle Detection', processed_frame)

    frame_count += 1
    print(f"Processing frame {frame_count} | Vehicle Count: {vehicle_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
control_led("led1 off")  # Turn off all LEDs
control_led("led2 off")
control_led("led3 off")
arduino.close()
