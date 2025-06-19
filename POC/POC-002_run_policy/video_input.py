import cv2

class VideoInput:
    
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.frameId = 0

        if not self.cap.isOpened():
            raise "Video not opened"


    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            raise "Video End"
        
        # Convert BGR (OpenCV) to RGB (matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb