Smart Traffic Light System using YOLOv5 and Arduino
===================================================

This project uses YOLOv5 and OpenCV to detect vehicles in a specific region of a video
and automatically control a traffic light system (connected via Arduino) based on vehicle count.
It helps optimize signal changes by detecting traffic density.

Features:
---------
- Real-time vehicle detection using a rotated Region of Interest (ROI)
- Dynamic traffic light control based on vehicle count
- Integration with Arduino via serial communication
- YOLOv5 object detection on CPU or GPU
- Customizable logic for LED (traffic light) behavior

Requirements:
-------------
- torch
- torchvision
- torchaudio
- opencv-python
- numpy
- pyserial
 Download YOLOv5 Model
You can download the pretrained yolov5s.pt from:

https://github.com/ultralytics/yolov5

Place the file in the project root directory.

Installation:
-------------
1. Install Python libraries:
   pip install -r requirements.txt

2. Connect your Arduino to the correct COM port.
   Update the arduino_port variable in the Python script.

3. Place the yolov5s.pt model in the same folder.

4. Run the main script:
   python main.py

Arduino LED Mapping:
--------------------
| LED | Color  | Command   |
|-----|--------|-----------|
| 1   | Green  | led1 on   |
| 2   | Yellow | led2 on   |
| 3   | Red    | led3 on   |

Logic:
------
- If vehicle count >= 3, turns green light on.
- If vehicle count < 3, turns red light on.
- Brief yellow light transition shown between state changes.

Usage:
------
Press 'Q' to quit the video window.
 Input
Tested with highway.mp4 video.
cap = cv2.VideoCapture('highway.mp4') // replace highway.mp4 with your video or use 0 for primary webcam and 1 for secondary webcam


Author: Kiran Bhusal
GitHub: https://github.com/kiranbhusal634
