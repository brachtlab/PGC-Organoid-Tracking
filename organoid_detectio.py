import cv2
import numpy as np
import json
from collections import deque

class OrganoidTracker:
    def __init__(self, video_path, total_frames=17140):
        self.video_path = video_path
        self.max_frames = total_frames
        self.frame_interval = 120
        self.circles = []
        self.tracking_data = {}
        self.temp_center = None
        self.selected_circle = None
        self.moving_circle = False
        self.capture = cv2.VideoCapture(video_path)
        self.current_frame = 0
        self.current_mouse_pos = None
        
    def select_circle(self, event, x, y, flags, param):
        self.current_mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.circles:
                for i, (cx, cy, r) in enumerate(self.circles):
                    if np.sqrt((x - cx)**2 + (y - cy)**2) < r:
                        self.selected_circle = i
                        self.moving_circle = True
                        return
            
            if self.temp_center is None:
                self.temp_center = (x, y)
            else:
                radius = np.round(np.sqrt((x - self.temp_center[0])**2 + (y - self.temp_center[1])**2))
                self.circles.append((self.temp_center[0], self.temp_center[1], radius))
                self.temp_center = None
                self.tracking_data[self.current_frame] = self.circles.copy()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving_circle and self.selected_circle is not None:
                self.circles[self.selected_circle] = (x, y, self.circles[self.selected_circle][2])
                self.tracking_data[self.current_frame] = self.circles.copy()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.moving_circle = False
            self.selected_circle = None
    
    def get_previous_frame_circles(self):
        prev_frames = [f for f in self.tracking_data.keys() if f < self.current_frame]
        if prev_frames:
            closest_frame = max(prev_frames)
            return self.tracking_data[closest_frame].copy()
        return []
    
    def track_circles(self):
        cv2.namedWindow('Organoid Tracking')
        cv2.setMouseCallback('Organoid Tracking', self.select_circle)
        
        while True:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.capture.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # If current frame has no circles, get them from previous frame
            if self.current_frame not in self.tracking_data:
                self.circles = self.get_previous_frame_circles()
                self.tracking_data[self.current_frame] = self.circles.copy()
            
            # Draw circles
            for x, y, r in self.circles:
                cv2.circle(display_frame, (int(x), int(y)), int(r), (0, 0, 255), 2)
            
            if self.temp_center is not None and self.current_mouse_pos is not None:
                cv2.circle(display_frame, self.temp_center, 3, (0, 255, 0), -1)
                radius = np.sqrt((self.current_mouse_pos[0] - self.temp_center[0])**2 + 
                               (self.current_mouse_pos[1] - self.temp_center[1])**2)
                cv2.circle(display_frame, self.temp_center, int(radius), (0, 255, 0), 1)
            
            cv2.putText(display_frame, f'Frame: {self.current_frame}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Organoid Tracking', display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                with open('organoid_tracking.json', 'w') as f:
                    json.dump(self.tracking_data, f)
                print(f"Saved tracking data for {len(self.tracking_data)} frames")
                self.current_frame = min(self.current_frame + self.frame_interval, self.max_frames - 1)
                self.circles = self.tracking_data.get(self.current_frame, [])
            elif key == ord('c'):
                if self.current_frame in self.tracking_data:
                    del self.tracking_data[self.current_frame]
                self.circles = []
        
        self.capture.release()
        cv2.destroyAllWindows()

def main():
    video_path = "../../data/T2_Video26.mp4"
    tracker = OrganoidTracker(video_path)
    tracker.track_circles()

if __name__ == "__main__":
    main()