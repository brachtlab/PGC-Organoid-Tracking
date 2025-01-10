"""
Cell Detection and Tracking System
--------------------------------------------
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from skimage.morphology import skeletonize
from scipy import ndimage
from collections import deque
import json
import pandas as pd

class Config:
    """Configuration parameters for the tracking system."""
    # ROI padding from edges
    ROI_PADDING = 10  # pixels from each edge
    
    # Video Processing
    PROCESSING_DURATION = 692  # seconds
    VIDEO_FPS = 24
    SLOW_MOTION_FACTOR = 0.4  
    
    # Object Detection
    SINGLE_CELL_THRESHOLD = 35
    CLUSTER_THRESHOLD = 210
    BLUR_LEVEL = 9
    BRIGHTNESS_THRESHOLD = 165
    CIRCULARITY_THRESHOLD = 0.25
    
    # Adaptive Thresholding
    SEARCH_RADIUS = 16  # pixels to search around last known position
    THRESHOLD_REDUCTION = 25  # how much to reduce threshold for known cells
    ADAPTIVE_WINDOW_SIZE = 35  # size of window for adaptive threshold
    
    # Image Processing
    KERNEL_SIZE = (5, 5)
    MIN_CONTRAST_VALUE = 170
    MAX_CONTRAST_VALUE = 190
    BRIDGE_WIDTH = 15
    
    # Tracking
    TRACKING_HISTORY = 600
    MAX_MOVEMENT_THRESHOLD = 9
    OBJECT_PERSISTENCE_THRESHOLD = 7
    LOST_OBJECT_PATIENCE = 9
    
    # Frame Averaging
    FRAME_BUFFER_SIZE = 7
    MIN_FRAME_DETECTIONS = 4
    POSITION_AVERAGING_WEIGHT = 0.7
    
    # Period Analysis
    FRAMES_PER_PERIOD = 120  
    MIN_TRACK_LENGTH = 40    
    
    # Grid Configuration
    GRID_COLS = 15
    GRID_ROWS = 10
    
    # Multiplier for determining near zone threshold
    CIRCLE_RADIUS_MULTIPLIER = 3.0

class ObjectTracker:
    """Tracks individual objects (cells or clusters) across frames."""
    
    def __init__(self, object_type, object_id, initial_position, initial_area, initial_contour, roi):
        self.object_type = object_type
        self.object_id = object_id
        self.positions = deque(maxlen=Config.TRACKING_HISTORY)
        self.areas = deque(maxlen=Config.TRACKING_HISTORY)
        self.contours = deque(maxlen=Config.TRACKING_HISTORY)
        self.frames_tracked = 1
        self.lost_frames = 0
        self.velocity = np.array([0.0, 0.0])
        self.roi = roi  # Store ROI for position prediction
        
        self.positions.append(initial_position)
        self.areas.append(initial_area)
        self.contours.append(initial_contour)
    
    def update(self, new_position, new_area, new_contour):
        """Update object with new detection data."""
        if len(self.positions) >= 1:
            old_pos = np.array(self.positions[-1])
            new_pos = np.array(new_position)
            self.velocity = new_pos - old_pos
        
        self.positions.append(new_position)
        self.areas.append(new_area)
        self.contours.append(new_contour)
        self.frames_tracked += 1
        self.lost_frames = 0
    
    def predict_next_position(self):
        """Predict next position based on current trajectory."""
        if len(self.positions) < 2:
            return self.positions[-1]
        
        current_pos = np.array(self.positions[-1])
        predicted = current_pos + self.velocity
        
        # Use instance ROI instead of Config.ROI
        predicted[0] = np.clip(predicted[0], self.roi[0], self.roi[0] + self.roi[2])
        predicted[1] = np.clip(predicted[1], self.roi[1], self.roi[1] + self.roi[3])
        
        return tuple(map(int, predicted))
    
    def get_smoothed_position(self):
        """Calculate smoothed position using weighted average."""
        if len(self.positions) < 2:
            return self.positions[-1]
        
        positions = np.array(self.positions)
        weights = np.linspace(0.5, 1.0, len(positions))
        weighted_pos = np.average(positions, axis=0, weights=weights)
        return tuple(map(int, weighted_pos))

class CellTracker:
    """Main class for cell and cluster tracking in video."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.output_dir = self._create_output_directory()
        self.next_object_id = 0
        self.frame_buffer = deque(maxlen=3)
        self.detection_buffer = deque(maxlen=3)
        self.roi = self._initialize_roi()
        self.organoid_data = self._load_organoid_data()
        
        self.tracking_stats = {}
        self.historical_tracks = []
        
        self.histogram_data = {
            'near': [],    # Combined data for zones 1,2
            'far': [],     # Combined data for zones 4,5
            'zone1': [],   # Individual zone data
            'zone2': [],
            'zone4': [],
            'zone5': []
        }
        self.period_histograms = []  # Store each period's histogram data
        self.current_near_zone_threshold = None 
    
    def _initialize_roi(self):
        """Initialize ROI based on video dimensions."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        # Get frame dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Calculate ROI coordinates
        x = Config.ROI_PADDING
        y = Config.ROI_PADDING
        w = width - (2 * Config.ROI_PADDING)
        h = height - (2 * Config.ROI_PADDING)
        
        # Ensure ROI dimensions are positive
        if w <= 0 or h <= 0:
            raise ValueError("ROI padding is too large for the video dimensions")
        
        return (x, y, w, h)
    
    def _save_tracking_statistics(self):
        """Save tracking statistics to CSV."""
        if not self.tracking_stats:
            print("No tracking statistics to save.")
            return
        
        print(f"Found {len(self.tracking_stats)} tracked objects to save.")
        
        # Save tracking data to CSV
        csv_path = os.path.join(self.output_dir, "tracking_statistics.csv")
        
        # Prepare data for CSV
        data = []
        for obj_id, stats in self.tracking_stats.items():
            if stats['total_frames'] >= Config.OBJECT_PERSISTENCE_THRESHOLD:
                data.append({
                    'object_id': obj_id,
                    'object_type': stats['object_type'],
                    'first_frame': stats['first_frame'],
                    'last_frame': stats['last_frame'],
                    'frames_tracked': stats['total_frames']
                })
        
        # Sort by object ID
        data.sort(key=lambda x: x['object_id'])
        
        # Write to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        print(f"Tracking statistics saved to: {csv_path}")
    
    def _create_adaptive_threshold_mask(self, frame_shape, tracked_objects):
        """Create a mask for adaptive thresholding based on tracked objects."""
        mask = np.zeros(frame_shape, dtype=np.uint8)
        
        if tracked_objects:
            for obj in tracked_objects:
                if obj.object_type == "cell" and len(obj.positions) > 0:
                    last_pos = obj.positions[-1]
                    # Convert global coordinates to ROI coordinates
                    roi_pos = (
                        int(last_pos[0] - self.roi[0]),
                        int(last_pos[1] - self.roi[1])
                    )
                    
                    # Ensure position is within frame bounds
                    if (0 <= roi_pos[1] < frame_shape[0] and 
                        0 <= roi_pos[0] < frame_shape[1]):
                        
                        # Calculate region bounds
                        y_start = max(0, roi_pos[1] - Config.SEARCH_RADIUS)
                        y_end = min(frame_shape[0], roi_pos[1] + Config.SEARCH_RADIUS + 1)
                        x_start = max(0, roi_pos[0] - Config.SEARCH_RADIUS)
                        x_end = min(frame_shape[1], roi_pos[0] + Config.SEARCH_RADIUS + 1)
                        
                        # Create circular mask for this region
                        region_height = y_end - y_start
                        region_width = x_end - x_start
                        
                        if region_height > 0 and region_width > 0:
                            y, x = np.ogrid[-region_height//2:region_height//2 + region_height%2,
                                          -region_width//2:region_width//2 + region_width%2]
                            disk = x*x + y*y <= Config.SEARCH_RADIUS*Config.SEARCH_RADIUS
                            
                            # Apply the disk mask to the region
                            mask[y_start:y_end, x_start:x_end][disk[:region_height, :region_width]] = Config.THRESHOLD_REDUCTION
        
        return mask

    def _apply_adaptive_threshold(self, gray_image, threshold_mask):
        """Apply adaptive thresholding with different levels for tracked cells."""
        # Ensure shapes match
        if gray_image.shape != threshold_mask.shape:
            raise ValueError(f"Shape mismatch: gray_image {gray_image.shape} != threshold_mask {threshold_mask.shape}")
        
        # Initialize output array
        thresh = np.zeros_like(gray_image, dtype=np.uint8)
        
        # Apply base threshold everywhere
        base_mask = gray_image > Config.BRIGHTNESS_THRESHOLD
        thresh[base_mask] = 255
        
        # Apply reduced threshold in tracked areas
        reduced_areas = threshold_mask > 0
        if np.any(reduced_areas):
            local_threshold = Config.BRIGHTNESS_THRESHOLD - threshold_mask[reduced_areas]
            local_mask = gray_image[reduced_areas] > local_threshold
            thresh[reduced_areas] = np.where(local_mask, 255, thresh[reduced_areas])
        
        return thresh
    
    def _create_output_directory(self):
        """Create timestamped output directory."""
        base_dir = "cell_tracking_output"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _preprocess_frame(self, frame, tracked_objects=None):
        """Preprocess video frame for detection with adaptive thresholding."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, Config.BLUR_LEVEL)
        
        # Contrast stretching
        stretched = np.clip(blurred, Config.MIN_CONTRAST_VALUE, Config.MAX_CONTRAST_VALUE)
        stretched = ((stretched - Config.MIN_CONTRAST_VALUE) / 
                    (Config.MAX_CONTRAST_VALUE - Config.MIN_CONTRAST_VALUE) * 255).astype(np.uint8)
        
        # Create adaptive threshold mask if we have tracked objects
        if tracked_objects:
            try:
                threshold_mask = self._create_adaptive_threshold_mask(stretched.shape, tracked_objects)
                thresh = self._apply_adaptive_threshold(stretched, threshold_mask)
            except Exception as e:
                print(f"Warning: Adaptive thresholding failed ({str(e)}), falling back to standard threshold")
                _, thresh = cv2.threshold(stretched, Config.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        else:
            # Use standard thresholding for first frame
            _, thresh = cv2.threshold(stretched, Config.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Clean up the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.KERNEL_SIZE)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _detect_and_classify_objects(self, binary_image, roi):
        """Detect and classify objects in preprocessed frame."""
        # Break bridges between objects
        separated = self._break_bridges(binary_image)
        
        # Find contours
        contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= Config.SINGLE_CELL_THRESHOLD:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) + roi[0]
                    cy = int(M['m01'] / M['m00']) + roi[1]
                    contour_shifted = cnt + np.array([roi[0], roi[1]])
                    
                    # Prioritize cell classification
                    if circularity > Config.CIRCULARITY_THRESHOLD and area < Config.CLUSTER_THRESHOLD:
                        objects.append(("cell", (cx, cy), area, contour_shifted))
                    elif area >= Config.CLUSTER_THRESHOLD:
                        # Additional check for potential multi-cell clusters
                        if circularity > Config.CIRCULARITY_THRESHOLD * 0.7:
                            # Treat as multiple cells if shape is relatively circular
                            objects.append(("cell", (cx, cy), area, contour_shifted))
                        else:
                            objects.append(("cluster", (cx, cy), area, contour_shifted))
        
        return objects
    
    def _break_bridges(self, binary_image):
        """Break narrow bridges between objects."""
        skeleton = skeletonize(binary_image > 0)
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        bridge_points = np.zeros_like(skeleton, dtype=np.uint8)
        bridge_points[skeleton & (dist_transform < Config.BRIDGE_WIDTH/2)] = 1
        
        result = binary_image.copy()
        result[bridge_points > 0] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def _match_and_update_tracks(self, current_objects, tracked_objects):
        """Match detected objects with existing tracks."""
        if not tracked_objects:
            return [ObjectTracker(obj_type, self.next_object_id + i, pos, area, contour, self.roi)
                    for i, (obj_type, pos, area, contour) in enumerate(current_objects)]
        
        matched_tracks = []
        unmatched_detections = list(range(len(current_objects)))
        
        for tracked_obj in tracked_objects:
            predicted_pos = tracked_obj.predict_next_position()
            best_match = None
            min_dist = float('inf')
            
            for i in unmatched_detections:
                obj_type, pos, area, contour = current_objects[i]
                if obj_type == tracked_obj.object_type:
                    dist = np.sqrt((predicted_pos[0] - pos[0])**2 + 
                                (predicted_pos[1] - pos[1])**2)
                    
                    if dist < Config.MAX_MOVEMENT_THRESHOLD and dist < min_dist:
                        min_dist = dist
                        best_match = i
            
            if best_match is not None:
                obj_type, pos, area, contour = current_objects[best_match]
                tracked_obj.update(pos, area, contour)
                matched_tracks.append(tracked_obj)
                unmatched_detections.remove(best_match)
            else:
                tracked_obj.lost_frames += 1
                if tracked_obj.lost_frames < Config.LOST_OBJECT_PATIENCE:
                    matched_tracks.append(tracked_obj)
        
        for i in unmatched_detections:
            obj_type, pos, area, contour = current_objects[i]
            new_track = ObjectTracker(obj_type, self.next_object_id, pos, area, contour, self.roi)
            self.next_object_id += 1
            matched_tracks.append(new_track)
        
        return matched_tracks
    
    def _load_organoid_data(self):
        """Load organoid tracking data from file with correct frame number handling."""
        try:
            with open('organoid_tracking.json', 'r') as f:
                data = json.load(f)
                # Convert the data to use integer keys and ensure proper format
                formatted_data = {}
                for k, v in data.items():
                    frame_num = int(k)
                    formatted_data[frame_num] = [[float(x), float(y), float(r)] for x, y, r in v]
                print(f"Successfully loaded organoid data with {len(formatted_data)} periods")
                return formatted_data
        except FileNotFoundError:
            print("Warning: organoid_tracking.json not found in current directory")
            return {}
        except Exception as e:
            print(f"Error loading organoid data: {str(e)}")
            return {}

    def _interpolate_organoid_circles(self, frame_number):
        """Interpolate organoid circles between known frames with debug prints."""
        
        if not self.organoid_data:
            print("No organoid data available!")
            return None
            
        # Find the surrounding known frames
        prev_period = (frame_number // Config.FRAMES_PER_PERIOD) * Config.FRAMES_PER_PERIOD
        next_period = prev_period + Config.FRAMES_PER_PERIOD
        
        # If we don't have both surrounding frames, use the closest one we have
        if prev_period not in self.organoid_data and next_period not in self.organoid_data:
            print(f"Neither {prev_period} nor {next_period} found in data")
            available_frames = list(self.organoid_data.keys())
            if not available_frames:
                return None
            closest_frame = min(available_frames, key=lambda x: abs(x - frame_number))
            print(f"Using closest frame: {closest_frame}")
            return self.organoid_data[closest_frame]
            
        if prev_period not in self.organoid_data:
            print(f"Using next period frame: {next_period}")
            return self.organoid_data[next_period]
        if next_period not in self.organoid_data:
            print(f"Using prev period frame: {prev_period}")
            return self.organoid_data[prev_period]
            
        # We have both frames, interpolate between them
        prev_circles = self.organoid_data[prev_period]
        next_circles = self.organoid_data[next_period]
        
        # Calculate interpolation factor
        alpha = (frame_number - prev_period) / Config.FRAMES_PER_PERIOD
        
        # Interpolate each circle
        interpolated_circles = []
        for prev_circle, next_circle in zip(prev_circles, next_circles):
            x = prev_circle[0] + alpha * (next_circle[0] - prev_circle[0])
            y = prev_circle[1] + alpha * (next_circle[1] - prev_circle[1])
            r = prev_circle[2] + alpha * (next_circle[2] - prev_circle[2])
            interpolated_circles.append([x, y, r])
            
        return interpolated_circles

    def _create_visualization(self, frame, tracked_objects):
        """Create visualization frame with updated organoid circles."""
        vis_frame = frame.copy()
        
        # Get current frame number from the tracked objects
        # Assuming objects maintain their frame count relative to the start
        current_frame = 0
        if tracked_objects and len(tracked_objects) > 0:
            # Find the most recently tracked object
            max_frames = max(obj.frames_tracked for obj in tracked_objects)
            current_frame = max_frames - 1  # Convert to 0-based frame number
        
        # Get and draw interpolated organoid circles for the current frame
        organoid_circles = self._interpolate_organoid_circles(current_frame)
        if organoid_circles:
            for circle in organoid_circles:
                x, y, r = circle
                # Draw circle outline
                cv2.circle(vis_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # Draw center point
                cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw tracked objects
        if tracked_objects:
            for obj in tracked_objects:
                if obj.frames_tracked >= Config.OBJECT_PERSISTENCE_THRESHOLD:
                    color = (0, 0, 255) if obj.object_type == "cell" else (255, 0, 0)
                    cv2.drawContours(vis_frame, [obj.contours[-1]], -1, color, 2)
        
        # Add frame counter to visualization
        cv2.putText(vis_frame, f'Frame: {current_frame}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return vis_frame

    def _create_trajectory_visualization(self, frame_size, frame_number, historical_tracks):
        """Create visualization showing trajectories and interpolated organoid circles."""
        trajectory_frame = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)
        
        # Get interpolated organoid circles for the current frame
        organoid_circles = self._interpolate_organoid_circles(frame_number)
        
        # Draw organoid circles if available
        if organoid_circles:
            for circle in organoid_circles:
                x, y, r = circle
                cv2.circle(trajectory_frame, (int(x), int(y)), int(r), (0, 255, 0), 1)
        
        # Rest of the trajectory visualization code...
        long_tracked_cells = [track for track in historical_tracks 
                            if track['total_frames'] >= 30 and
                            track['first_frame'] <= frame_number]
        
        for track in long_tracked_cells:
            if frame_number > track['last_frame']:
                continue
                
            positions = track['positions']
            max_idx = min(frame_number - track['first_frame'], 
                        len(positions),
                        track['last_frame'] - track['first_frame'])
            current_positions = positions[:max_idx]
            
            if len(current_positions) > 1:
                for i in range(1, len(current_positions)):
                    pt1 = tuple(map(int, current_positions[i-1]))
                    pt2 = tuple(map(int, current_positions[i]))
                    
                    progress = i / len(current_positions)
                    color = (
                        int(255 * progress),
                        0,
                        int(255 * (1-progress))
                    )
                    
                    cv2.line(trajectory_frame, pt1, pt2, color, 2)
                
                if current_positions and frame_number <= track['last_frame']:
                    current_pos = tuple(map(int, current_positions[-1]))
                    cv2.circle(trajectory_frame, current_pos, 3, (0, 255, 0), -1)
        
        cv2.putText(trajectory_frame, f'Frame: {frame_number}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return trajectory_frame
    
    def _average_detections(self, detections_list):
        """Average object detections across multiple frames."""
        if not detections_list:
            return []
            
        averaged_objects = []
        for frame_idx, frame_detections in enumerate(detections_list):
            for obj_type, pos, area, contour in frame_detections:
                matched = False
                
                for avg_obj in averaged_objects:
                    avg_type, avg_pos, avg_areas, avg_contours, count = avg_obj
                    
                    if obj_type == avg_type:
                        dist = np.sqrt((pos[0] - avg_pos[0])**2 + (pos[1] - avg_pos[1])**2)
                        if dist < Config.MAX_MOVEMENT_THRESHOLD:
                            new_x = (avg_pos[0] * count + pos[0]) / (count + 1)
                            new_y = (avg_pos[1] * count + pos[1]) / (count + 1)
                            avg_obj[1] = (int(new_x), int(new_y))
                            avg_obj[2].append(area)
                            avg_obj[3].append(contour)
                            avg_obj[4] += 1
                            matched = True
                            break
                
                if not matched:
                    averaged_objects.append([
                        obj_type,
                        pos,
                        [area],
                        [contour],
                        1
                    ])
        
        final_objects = []
        for avg_obj in averaged_objects:
            obj_type, pos, areas, contours, count = avg_obj
            if count >= 2:
                median_area = np.median(areas)
                recent_contour = contours[-1]
                final_objects.append((obj_type, pos, median_area, recent_contour))
        
        return final_objects
    
    def _record_detection(self, frame_number, tracked_objects):
        """Record tracking data including positions list and update tracking stats."""
        for obj in tracked_objects:
            if obj.frames_tracked >= Config.OBJECT_PERSISTENCE_THRESHOLD:
                # Update historical tracks
                track_data = {
                    'cell_id': obj.object_id,
                    'object_type': obj.object_type,
                    'first_frame': frame_number - obj.frames_tracked + 1,
                    'last_frame': frame_number,
                    'total_frames': obj.frames_tracked,
                    'positions': list(obj.positions),
                    'lost_frames': obj.lost_frames
                }
                
                # Update tracking stats
                self.tracking_stats[obj.object_id] = {
                    'object_type': obj.object_type,
                    'first_frame': track_data['first_frame'],
                    'last_frame': track_data['last_frame'],
                    'total_frames': obj.frames_tracked
                }
                
                # Update historical tracks
                existing = next((item for item in self.historical_tracks 
                            if item['cell_id'] == obj.object_id), None)
                if existing:
                    existing.update(track_data)
                else:
                    self.historical_tracks.append(track_data)

    def _get_near_zone_threshold(self, frame_number):
        """Calculate dynamic near zone threshold based on circle radius."""
        circles = self._interpolate_organoid_circles(frame_number)
            
        # Get the radius from the first circle and multiply by 3
        radius = circles[0][2]  # circles contain [x, y, radius]
        return radius * Config.CIRCLE_RADIUS_MULTIPLIER
    
    def _classify_grid_zones(self, frame_number):
        """Classify grid cells into zones based on organoid circle positions."""
        cell_width = self.roi[2] // Config.GRID_COLS
        cell_height = self.roi[3] // Config.GRID_ROWS
        
        zones = np.zeros((Config.GRID_ROWS, Config.GRID_COLS), dtype=int)
        
        circles = self._interpolate_organoid_circles(frame_number)
        if not circles:
            print("No circles found")
            return zones
        
        # Convert circle coordinates to ROI coordinates
        roi_circles = []
        for circle in circles:
            x, y, r = circle
            roi_x = x - self.roi[0]
            roi_y = y - self.roi[1]
            roi_circles.append((roi_x, roi_y, r))
        
        circle_center = np.array([roi_circles[0][0], roi_circles[0][1]])
        
        # Get dynamic threshold for this frame
        near_zone_threshold = self._get_near_zone_threshold(frame_number)
        self.current_near_zone_threshold = near_zone_threshold  # Store for visualization
        
        # Classify each grid cell using dynamic threshold
        for y in range(Config.GRID_ROWS):
            for x in range(Config.GRID_COLS):
                cell_center_x = (x + 0.5) * cell_width
                cell_center_y = (y + 0.5) * cell_height
                cell_point = np.array([cell_center_x, cell_center_y])
                
                dist_to_circle = np.linalg.norm(cell_point - circle_center)
                
                if dist_to_circle < near_zone_threshold:
                    zones[y, x] = 1  # Near circle
                else:
                    zones[y, x] = 4  # Far area
        
        # Print statistics
        unique, counts = np.unique(zones, return_counts=True)
        zone_counts = dict(zip(unique, counts))
        print("\nZone classification statistics:")
        print(f"Near circle (1): {zone_counts.get(1, 0)} cells")
        print(f"Far area (4): {zone_counts.get(4, 0)} cells")
        print(f"Current near zone threshold: {near_zone_threshold:.1f}")
        
        return zones

    def _create_grid_visualization(self, frame_size, frame_number, historical_tracks):
        """Create grid visualization with updated zone coloring and movement vectors."""
        if frame_number < Config.FRAMES_PER_PERIOD or frame_number % Config.FRAMES_PER_PERIOD != 0:
            return None
        
        window_start = frame_number - Config.FRAMES_PER_PERIOD
        window_end = frame_number
        period = window_end // Config.FRAMES_PER_PERIOD
        
        print(f"\nCreating grid visualization for frame {frame_number} (Period {period})")
        
        vis_frame = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)
        
        # Get zone classification
        zones = self._classify_grid_zones(frame_number)
        
        cell_width = self.roi[2] // Config.GRID_COLS
        cell_height = self.roi[3] // Config.GRID_ROWS
        
        # Define colors for each zone - simplified for single circle
        zone_colors = {
            1: (255, 200, 200),  # Light red for near circle
            4: (200, 255, 200),  # Light green for far area
        }
        
        # Draw zone backgrounds
        for y in range(Config.GRID_ROWS):
            for x in range(Config.GRID_COLS):
                zone = zones[y, x]
                if zone in zone_colors:
                    x1 = self.roi[0] + x * cell_width
                    y1 = self.roi[1] + y * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    color = zone_colors[zone]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, -1)

        # Initialize grid movement storage with zones
        grid_movements = {}
        for y in range(Config.GRID_ROWS):
            for x in range(Config.GRID_COLS):
                grid_movements[(x, y)] = {
                    'movements': [],
                    'zone': zones[y, x]
                }
        
        # Process tracks within current window
        for track in historical_tracks:
            if (track['first_frame'] <= window_end and 
                track['last_frame'] >= window_start and 
                track['object_type'] == 'cell' and
                track['total_frames'] >= Config.MIN_TRACK_LENGTH):
                
                window_positions = []
                positions = track['positions']
                
                for i, pos in enumerate(positions):
                    frame_idx = track['first_frame'] + i
                    if window_start <= frame_idx <= window_end:
                        window_positions.append(pos)
                
                if len(window_positions) >= 2:
                    end_pos = np.array(window_positions[-1])
                    start_pos = np.array(window_positions[0])
                    movement = end_pos - start_pos
                    
                    roi_x = end_pos[0] - self.roi[0]
                    roi_y = end_pos[1] - self.roi[1]
                    grid_x = int(np.clip(roi_x // cell_width, 0, Config.GRID_COLS - 1))
                    grid_y = int(np.clip(roi_y // cell_height, 0, Config.GRID_ROWS - 1))
                    grid_key = (grid_x, grid_y)
                    
                    grid_movements[grid_key]['movements'].append(movement)
        
        # Draw organoid circles
        if frame_number in self.organoid_data:
            circles = self.organoid_data[frame_number]
            for circle in circles:
                x, y, r = circle
                cv2.circle(vis_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw movement vectors
        vector_colors = {
            1: (200, 0, 0),    # Red for near circle
            4: (0, 150, 0)     # Dark green for far area
        }
        
        for grid_key, data in grid_movements.items():
            movements = data['movements']
            zone = data['zone']
            if movements and zone in vector_colors:  # Only draw if we have movements and valid zone
                x, y = grid_key
                start_point = (
                    int(self.roi[0] + x * cell_width + cell_width/2),
                    int(self.roi[1] + y * cell_height + cell_height/2)
                )
                
                avg_movement = np.mean(movements, axis=0)
                
                # Scale factor for vector visualization
                scale = 0.5
                end_point = (
                    int(start_point[0] + avg_movement[0] * scale),
                    int(start_point[1] + avg_movement[1] * scale)
                )
                
                # Draw arrow
                cv2.arrowedLine(vis_frame, start_point, end_point, vector_colors[zone], 2)
                
                # Add magnitude text
                magnitude = np.linalg.norm(avg_movement)
                cv2.putText(vis_frame, f"{magnitude:.1f}", 
                        (start_point[0], start_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, vector_colors[zone], 1)
        
        # Draw grid lines
        for x in range(Config.GRID_COLS + 1):
            x_pos = self.roi[0] + x * cell_width
            cv2.line(vis_frame, (x_pos, self.roi[1]), 
                    (x_pos, self.roi[1] + self.roi[3]), (100, 100, 100), 1)
        
        for y in range(Config.GRID_ROWS + 1):
            y_pos = self.roi[1] + y * cell_height
            cv2.line(vis_frame, (self.roi[0], y_pos),
                    (self.roi[0] + self.roi[2], y_pos), (100, 100, 100), 1)
        
        # Add legend
        legend_y = 60
        zone_names = {
            1: "Near Circle",
            4: "Far Area"
        }
        
        # Draw legend boxes and labels
        for zone, name in zone_names.items():
            # Background color box
            cv2.rectangle(vis_frame, (10, legend_y), (30, legend_y + 20), 
                        zone_colors[zone], -1)
            # Vector color example
            cv2.arrowedLine(vis_frame, (35, legend_y + 10), (55, legend_y + 10), 
                        vector_colors[zone], 2)
            # Zone name
            cv2.putText(vis_frame, name, (65, legend_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 30
        
        # Add threshold and frame information
        if self.current_near_zone_threshold is not None:
            cv2.putText(vis_frame, f'Near Zone Threshold: {self.current_near_zone_threshold:.1f}px ({Config.CIRCLE_RADIUS_MULTIPLIER}x radius)', 
                        (10, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(vis_frame, f'Frame: {frame_number} (Period {period})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return vis_frame
    
    def _log_period_statistics(self):
        """Log detection statistics for each time period and grid cells."""
        stats_path = os.path.join(self.output_dir, "detection_statistics.txt")
        
        # Calculate grid dimensions
        grid_cols = 15
        grid_rows = 10
        cell_width = self.roi[2] // grid_cols
        cell_height = self.roi[3] // grid_rows
        
        # Initialize statistics containers
        period_stats = {}  # Format: {period: set(cell_ids)}
        grid_stats = {}    # Format: {period: {grid_cell: set(cell_ids)}}
        
        # Process all historical tracks
        for track in self.historical_tracks:
            if track['total_frames'] >= 30:  # Only consider tracks lasting more than 30 frames
                cell_id = track['cell_id']
                start_frame = track['first_frame']
                end_frame = track['last_frame']
                
                # Calculate which periods this track appears in
                start_period = (start_frame // 120) + 1
                end_period = (end_frame // 120) + 1
                
                # Get the last position for grid cell calculation
                last_pos = np.array(track['positions'][-1])
                
                # Calculate grid position
                roi_x = last_pos[0] - self.roi[0]
                roi_y = last_pos[1] - self.roi[1]
                grid_x = int(np.clip(roi_x // cell_width, 0, grid_cols - 1))
                grid_y = int(np.clip(roi_y // cell_height, 0, grid_rows - 1))
                grid_key = (grid_x, grid_y)
                
                # Record statistics for each period the track appears in
                for period in range(start_period, end_period + 1):
                    # Initialize period if not exists
                    if period not in period_stats:
                        period_stats[period] = set()
                        grid_stats[period] = {}
                    
                    # Add to period stats
                    period_stats[period].add(cell_id)
                    
                    # Initialize grid cell if not exists
                    if grid_key not in grid_stats[period]:
                        grid_stats[period][grid_key] = set()
                    
                    # Add to grid stats
                    grid_stats[period][grid_key].add(cell_id)
        
        # Write statistics to file
        with open(stats_path, 'w') as f:
            f.write("Cell Detection Statistics\n")
            f.write("=======================\n\n")
            
            # Overall statistics by period
            f.write("Period Statistics (>= 30 frames)\n")
            f.write("--------------------------------\n")
            total_unique_cells = set()
            for period in sorted(period_stats.keys()):
                frame_start = (period - 1) * 120 + 1
                frame_end = period * 120
                cells_in_period = period_stats[period]
                total_unique_cells.update(cells_in_period)
                
                f.write(f"Period {period} (Frames {frame_start}-{frame_end}): {len(cells_in_period)} cells\n")
                f.write(f"Cell IDs: {sorted(cells_in_period)}\n\n")
            
            f.write(f"Total unique cells across all periods: {len(total_unique_cells)}\n")
            f.write(f"All unique cell IDs: {sorted(total_unique_cells)}\n\n")
            
            # Grid statistics by period
            f.write("\nGrid Cell Statistics\n")
            f.write("-------------------\n")
            for period in sorted(grid_stats.keys()):
                frame_start = (period - 1) * 120 + 1
                frame_end = period * 120
                f.write(f"\nPeriod {period} (Frames {frame_start}-{frame_end})\n")
                
                # Calculate total cells in all grid cells for this period
                total_grid_cells = set()
                for grid_key, cells in grid_stats[period].items():
                    total_grid_cells.update(cells)
                    
                    f.write(f"Grid ({grid_key[0]}, {grid_key[1]}): {len(cells)} cells\n")
                    f.write(f"  Cell IDs: {sorted(cells)}\n")
                
                # Verification
                f.write(f"\nVerification for Period {period}:\n")
                f.write(f"Sum of all grid cells: {len(total_grid_cells)} unique cells\n")
                f.write(f"Total in period stats: {len(period_stats[period])} cells\n")
                
                # Check if numbers match
                if len(total_grid_cells) != len(period_stats[period]):
                    f.write("WARNING: Mismatch between grid total and period total!\n")
                    missing_cells = period_stats[period] - total_grid_cells
                    extra_cells = total_grid_cells - period_stats[period]
                    if missing_cells:
                        f.write(f"Cells in period but not in grids: {sorted(missing_cells)}\n")
                    if extra_cells:
                        f.write(f"Cells in grids but not in period: {sorted(extra_cells)}\n")
                else:
                    f.write("Verification passed: Grid total matches period total\n")
                f.write("\n")
            
            # Summary statistics
            f.write("\nSummary Statistics\n")
            f.write("-----------------\n")
            f.write(f"Total number of periods: {len(period_stats)}\n")
            f.write(f"Average cells per period: {sum(len(cells) for cells in period_stats.values()) / len(period_stats):.2f}\n")
            f.write(f"Total unique cells tracked: {len(total_unique_cells)}\n")
            
            print(f"Statistics have been logged to: {stats_path}")
            
    def _analyze_track_periods(self):
        """Analyze which tracks span multiple periods vs stay within one period."""
        single_period_tracks = []
        multi_period_tracks = []
        
        for track in self.historical_tracks:
            if track['total_frames'] >= Config.MIN_TRACK_LENGTH:  # Only analyze significant tracks
                start_period = (track['first_frame'] // Config.FRAMES_PER_PERIOD) + 1
                end_period = (track['last_frame'] // Config.FRAMES_PER_PERIOD) + 1
                
                track_info = {
                    'object_id': track['cell_id'],
                    'object_type': track['object_type'],
                    'first_frame': track['first_frame'],
                    'last_frame': track['last_frame'],
                    'frames_tracked': track['total_frames'],
                    'start_period': start_period,
                    'end_period': end_period
                }
                
                if start_period == end_period:
                    single_period_tracks.append(track_info)
                else:
                    multi_period_tracks.append(track_info)
        
        # Sort tracks by object_id
        single_period_tracks.sort(key=lambda x: x['object_id'])
        multi_period_tracks.sort(key=lambda x: x['object_id'])
        
        # Write analysis to file
        analysis_path = os.path.join(self.output_dir, "track_period_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write("Track Period Analysis\n")
            f.write("===================\n\n")
            
            f.write("Single Period Tracks\n")
            f.write("-------------------\n")
            f.write("object_id | object_type | first_frame | last_frame | frames_tracked | period\n")
            for track in single_period_tracks:
                f.write(f"{track['object_id']:9d} | {track['object_type']:11s} | {track['first_frame']:11d} | "
                    f"{track['last_frame']:10d} | {track['frames_tracked']:13d} | {track['start_period']}\n")
            
            f.write("\nMulti-Period Tracks\n")
            f.write("-----------------\n")
            f.write("object_id | object_type | first_frame | last_frame | frames_tracked | start_period | end_period\n")
            for track in multi_period_tracks:
                f.write(f"{track['object_id']:9d} | {track['object_type']:11s} | {track['first_frame']:11d} | "
                    f"{track['last_frame']:10d} | {track['frames_tracked']:13d} | {track['start_period']:12d} | "
                    f"{track['end_period']}\n")
            
            # Summary statistics
            f.write(f"\nSummary:\n")
            f.write(f"Total tracks: {len(single_period_tracks) + len(multi_period_tracks)}\n")
            f.write(f"Single period tracks: {len(single_period_tracks)}\n")
            f.write(f"Multi-period tracks: {len(multi_period_tracks)}\n")
            
    def _analyze_track_continuity(self):
        """Analyze tracks crossing period boundaries with consistent table format."""
        period_boundaries = {}
        
        for track in self.historical_tracks:
            if track['total_frames'] >= 30:  # Only analyze significant tracks
                start_period = (track['first_frame'] // 120) + 1
                end_period = (track['last_frame'] // 120) + 1
                
                for period in range(start_period, end_period):
                    boundary_frame = period * 120
                    
                    if boundary_frame not in period_boundaries:
                        period_boundaries[boundary_frame] = []
                    
                    # Calculate frames at boundary
                    frames_passed = boundary_frame - track['first_frame']
                    frames_remaining = track['last_frame'] - boundary_frame
                    
                    # Determine case and frame counting
                    if frames_passed < 30:
                        case = 1
                        count_in_current = 0
                        count_in_next = track['total_frames']
                    elif frames_passed >= 30 and frames_remaining < 30:
                        case = 2
                        count_in_current = 0
                        count_in_next = track['total_frames']
                    else:  # frames_passed >= 30 and frames_remaining >= 30
                        case = 3
                        count_in_current = frames_passed
                        count_in_next = frames_remaining
                    
                    track_info = {
                        'object_id': track['cell_id'],
                        'object_type': track['object_type'],
                        'first_frame': track['first_frame'],
                        'last_frame': track['last_frame'],
                        'frames_passed': frames_passed,
                        'frames_remaining': frames_remaining,
                        'case': case,
                        'frames_in_current': count_in_current,
                        'frames_in_next': count_in_next
                    }
                    
                    period_boundaries[boundary_frame].append(track_info)
        
        # Write analysis to file
        continuity_path = os.path.join(self.output_dir, "track_continuity_analysis.txt")
        with open(continuity_path, 'w') as f:
            f.write("Track Continuity Analysis\n")
            f.write("========================\n\n")
            
            for boundary_frame in sorted(period_boundaries.keys()):
                tracks = period_boundaries[boundary_frame]
                if tracks:
                    period_i = boundary_frame // 120
                    f.write(f"\nPeriod Boundary at Frame {boundary_frame}\n")
                    f.write(f"Period {period_i} → {period_i + 1}\n")
                    f.write("\n")
                    
                    # Table header
                    f.write("object_id | object_type | first_frame | last_frame | frames_passed | frames_remaining | case | "
                        "frames in p({}) | frames in p({})\n".format(period_i, period_i + 1))
                    f.write("-" * 120 + "\n")
                    
                    # Sort tracks by object_id
                    for track in sorted(tracks, key=lambda x: x['object_id']):
                        f.write(f"{track['object_id']:9d} | "
                            f"{track['object_type']:11s} | "
                            f"{track['first_frame']:11d} | "
                            f"{track['last_frame']:10d} | "
                            f"{track['frames_passed']:12d} | "
                            f"{track['frames_remaining']:15d} | "
                            f"{track['case']:4d} | "
                            f"{track['frames_in_current']:14d} | "
                            f"{track['frames_in_next']:14d}\n")
                    
                    # Summary statistics
                    f.write("\nSummary:\n")
                    case_counts = {1: 0, 2: 0, 3: 0}
                    for track in tracks:
                        case_counts[track['case']] += 1
                    
                    f.write(f"Total tracks crossing: {len(tracks)}\n")
                    f.write(f"Case 1 (counted in next period): {case_counts[1]}\n")
                    f.write(f"Case 2 (counted in next period): {case_counts[2]}\n")
                    f.write(f"Case 3 (split between periods): {case_counts[3]}\n")
                    f.write("\n" + "=" * 120 + "\n")
            
            # Overall statistics
            f.write("\nOverall Statistics\n")
            f.write("=================\n")
            total_tracks = sum(len(tracks) for tracks in period_boundaries.values())
            total_boundaries = len(period_boundaries)
            if total_boundaries > 0:
                f.write(f"Total period boundaries: {total_boundaries}\n")
                f.write(f"Total track crossings: {total_tracks}\n")
                f.write(f"Average tracks per boundary: {total_tracks/total_boundaries:.1f}\n")

    
    def _calculate_vector_angles(self, grid_movements, frame_number):
        """Calculate angles between cell movement vectors and vectors to circle center."""
        angles = {}
        
        # Get circle positions
        circles = self._interpolate_organoid_circles(frame_number)
        if not circles:
            print("No circles found")
            return {}
        
        # Convert to ROI coordinates
        circle_center = np.array([
            circles[0][0] - self.roi[0],
            circles[0][1] - self.roi[1]
        ])
        
        cell_width = self.roi[2] // Config.GRID_COLS
        cell_height = self.roi[3] // Config.GRID_ROWS
        
        for grid_key, data in grid_movements.items():
            movements = data['movements']
            zone = data['zone']
            
            if movements:  # Process all movements
                x, y = grid_key
                cell_center = np.array([
                    x * cell_width + cell_width/2,
                    y * cell_height + cell_height/2
                ])
                
                # Calculate average movement vector
                avg_movement = np.mean(movements, axis=0)
                if np.all(avg_movement == 0):
                    continue
                    
                # Calculate vector pointing to circle center
                to_circle = circle_center - cell_center
                
                # Calculate angle between vectors
                dot_product = np.dot(avg_movement, to_circle)
                movement_mag = np.linalg.norm(avg_movement)
                circle_mag = np.linalg.norm(to_circle)
                
                if movement_mag == 0 or circle_mag == 0:
                    continue
                    
                cos_angle = dot_product / (movement_mag * circle_mag)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                angles[grid_key] = {
                    'angle': angle_deg,
                    'zone': zone,
                    'magnitude': movement_mag
                }
        
        return angles

    def _get_angle_bin_color(self, angle):
        """Map angle to a color based on 20-degree bins."""
        # Define bin edges (0, 20, 40, ..., 180)
        bin_edges = np.arange(0, 181, 20)
        
        # Define colors for each bin (9 bins)
        bin_colors = [
            (255, 0, 0),    # Red
            (255, 128, 0),  # Orange
            (255, 255, 0),  # Yellow
            (128, 255, 0),  # Light green
            (0, 255, 0),    # Green
            (0, 255, 255),  # Cyan
            (0, 128, 255),  # Light blue
            (0, 0, 255),    # Blue
            (128, 0, 255)   # Purple
        ]
        
        # Find which bin the angle belongs to
        bin_idx = np.digitize(angle, bin_edges) - 1
        # Clip to ensure we don't exceed our color list
        bin_idx = min(bin_idx, len(bin_colors) - 1)
        
        return bin_colors[bin_idx]

    def _create_angle_histogram(self, angles, frame_number):
        """Create histograms of angles for single circle configuration."""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        zones = [1, 4]  # Only near and far zones
        zone_names = {
            1: "Near Circle",
            4: "Far Area"
        }
        
        # Initialize bins
        bins = np.arange(0, 181, 20)  # 20-degree bins
        bin_centers = bins[:-1] + 10
        
        # Define colors for each bin
        bin_colors = [
            'red',
            'orange',
            'yellow',
            'lightgreen',
            'green',
            'cyan',
            'lightblue',
            'blue',
            'purple'
        ]
        
        # Collect angles for each zone
        zone_angles = {zone: [] for zone in zones}
        for grid_key, angle_data in angles.items():
            zone = angle_data['zone']
            if zone in zones:
                zone_angles[zone].append(angle_data['angle'])
        
        # Store current period's data
        period = frame_number // Config.FRAMES_PER_PERIOD
        period_data = {
            'period': period,
            'near': zone_angles[1],
            'far': zone_angles[4]
        }
        self.period_histograms.append(period_data)
        
        # Update the cumulative histogram data
        self.histogram_data['near'].extend(zone_angles[1])
        self.histogram_data['far'].extend(zone_angles[4])
        
        # Create histograms
        for idx, zone in enumerate(zones, 1):
            plt.subplot(2, 1, idx)
            if zone_angles[zone]:
                n, bins, patches = plt.hist(zone_angles[zone], bins=bins, edgecolor='black')
                
                # Color each bin
                for i in range(len(patches)):
                    patches[i].set_facecolor(bin_colors[min(i, len(bin_colors)-1)])
                
                plt.title(f"{zone_names[zone]} (n={len(zone_angles[zone])})")
                plt.xlabel("Angle (degrees)")
                plt.ylabel("Count")
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save histogram
        hist_path = os.path.join(self.output_dir, f"angle_histogram_period_{period}.png")
        plt.savefig(hist_path)
        plt.close()
        
        return hist_path

    def _create_aggregate_histogram(self):
        """Create an aggregate histogram for single circle configuration."""
        if not self.histogram_data['near']:  # Check if we have any data
            return
            
        fig = plt.figure(figsize=(15, 10))
        bins = np.arange(0, 181, 20)
        bin_colors = [
            'red',
            'orange',
            'yellow',
            'lightgreen',
            'green',
            'cyan',
            'lightblue',
            'blue',
            'purple'
        ]
        
        # Plot histograms
        plot_data = [
            ('near', 1, "Near Circle"),
            ('far', 2, "Far Area")
        ]
        
        for key, plot_num, title in plot_data:
            plt.subplot(2, 1, plot_num)
            if self.histogram_data[key]:
                n, bins, patches = plt.hist(self.histogram_data[key], bins=bins, edgecolor='black')
                for i in range(len(patches)):
                    patches[i].set_facecolor(bin_colors[min(i, len(bin_colors)-1)])
                plt.title(f"{title} - All Periods (n={len(self.histogram_data[key])})")
                plt.xlabel("Angle (degrees)")
                plt.ylabel("Count")
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save aggregate histogram
        hist_path = os.path.join(self.output_dir, "angle_histogram_aggregate.png")
        plt.savefig(hist_path)
        plt.close()
        
        # Save the numerical data
        data_path = os.path.join(self.output_dir, "histogram_data.json")
        with open(data_path, 'w') as f:
            json.dump({
                'bins': bins.tolist(),
                'data': self.histogram_data,
                'period_data': self.period_histograms
            }, f, indent=2)

    def _create_angle_grid_visualization(self, frame_size, frame_number, historical_tracks):
        """Create grid visualization showing angles between movement and circle vectors."""
        if frame_number < Config.FRAMES_PER_PERIOD or frame_number % Config.FRAMES_PER_PERIOD != 0:
            return None
        
        window_start = frame_number - Config.FRAMES_PER_PERIOD
        window_end = frame_number
        period = window_end // Config.FRAMES_PER_PERIOD
        
        print(f"\nCreating angle grid visualization for frame {frame_number} (Period {period})")
        
        vis_frame = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)
        
        # Get zone classification
        zones = self._classify_grid_zones(frame_number)
        
        cell_width = self.roi[2] // Config.GRID_COLS
        cell_height = self.roi[3] // Config.GRID_ROWS
        
        # Define colors for each zone
        zone_colors = {
            1: (255, 200, 200),  # Light red for near circle 1
            2: (200, 200, 255),  # Light blue for near circle 2
            3: (240, 240, 200),  # Light yellow for between circles
            4: (200, 255, 200),  # Light green for far area - circle 1
            5: (220, 255, 220)   # Slightly different green for far area - circle 2
        }
        
        # Draw zone backgrounds
        for y in range(Config.GRID_ROWS):
            for x in range(Config.GRID_COLS):
                zone = zones[y, x]
                if zone > 0:
                    x1 = self.roi[0] + x * cell_width
                    y1 = self.roi[1] + y * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    color = zone_colors[zone]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, -1)

        # Calculate movements and angles
        grid_movements = {}
        for y in range(Config.GRID_ROWS):
            for x in range(Config.GRID_COLS):
                grid_movements[(x, y)] = {
                    'movements': [],
                    'zone': zones[y, x]
                }
        
        # Process tracks
        for track in historical_tracks:
            if (track['first_frame'] <= window_end and 
                track['last_frame'] >= window_start and 
                track['object_type'] == 'cell' and
                track['total_frames'] >= Config.MIN_TRACK_LENGTH):
                
                window_positions = []
                positions = track['positions']
                
                for i, pos in enumerate(positions):
                    frame_idx = track['first_frame'] + i
                    if window_start <= frame_idx <= window_end:
                        window_positions.append(pos)
                
                if len(window_positions) >= 2:
                    end_pos = np.array(window_positions[-1])
                    start_pos = np.array(window_positions[0])
                    movement = end_pos - start_pos
                    
                    roi_x = end_pos[0] - self.roi[0]
                    roi_y = end_pos[1] - self.roi[1]
                    grid_x = int(np.clip(roi_x // cell_width, 0, Config.GRID_COLS - 1))
                    grid_y = int(np.clip(roi_y // cell_height, 0, Config.GRID_ROWS - 1))
                    grid_key = (grid_x, grid_y)
                    
                    grid_movements[grid_key]['movements'].append(movement)
        
        # Calculate angles
        angles = self._calculate_vector_angles(grid_movements, frame_number)
        
        # Create histogram if we have angles to analyze
        if angles:
            self._create_angle_histogram(angles, frame_number)
        
        # Draw angles with binned colors
        for grid_key, angle_data in angles.items():
            x, y = grid_key
            zone = angle_data['zone']
            angle = angle_data['angle']
            magnitude = angle_data['magnitude']
            
            if zone != 3:  # Skip between circles zone
                center_x = int(self.roi[0] + x * cell_width + cell_width/2)
                center_y = int(self.roi[1] + y * cell_height + cell_height/2)
                
                # Use new binned color function
                angle_color = self._get_angle_bin_color(angle)
                
                # Draw angle text
                cv2.putText(vis_frame, f"{angle:.1f}", 
                        (center_x - 20, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, angle_color, 1)
                
                # Optional: Draw magnitude below angle
                cv2.putText(vis_frame, f"m:{magnitude:.1f}", 
                        (center_x - 20, center_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, angle_color, 1)
        
        # Draw organoid circles
        if frame_number in self.organoid_data:
            circles = self.organoid_data[frame_number]
            for circle in circles:
                x, y, r = circle
                cv2.circle(vis_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw grid lines
        for x in range(Config.GRID_COLS + 1):
            x_pos = self.roi[0] + x * cell_width
            cv2.line(vis_frame, (x_pos, self.roi[1]), 
                    (x_pos, self.roi[1] + self.roi[3]), (100, 100, 100), 1)
        
        for y in range(Config.GRID_ROWS + 1):
            y_pos = self.roi[1] + y * cell_height
            cv2.line(vis_frame, (self.roi[0], y_pos),
                    (self.roi[0] + self.roi[2], y_pos), (100, 100, 100), 1)
        
        # Add frame info
        cv2.putText(vis_frame, f'Frame: {frame_number} (Period {period})', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return vis_frame


    def process_video(self):
        """Process video using sliding window approach with continuous tracking."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        frame_count = int(min(Config.PROCESSING_DURATION * Config.VIDEO_FPS, 
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Initialize video writers
        output_path = os.path.join(self.output_dir, "tracked_video.mp4")
        trajectory_path = os.path.join(self.output_dir, "trajectory_video.mp4")
        grid_path = os.path.join(self.output_dir, "grid_video.mp4")
        angle_grid_path = os.path.join(self.output_dir, "angle_grid_video.mp4")
        
        out = cv2.VideoWriter(output_path, fourcc, 
                            Config.VIDEO_FPS * Config.SLOW_MOTION_FACTOR,
                            frame_size)
        trajectory_out = cv2.VideoWriter(trajectory_path, fourcc,
                                    Config.VIDEO_FPS * Config.SLOW_MOTION_FACTOR,
                                    frame_size)
        grid_out = cv2.VideoWriter(grid_path, fourcc,
                                Config.VIDEO_FPS,
                                frame_size)
        angle_grid_out = cv2.VideoWriter(angle_grid_path, fourcc,        
                                Config.VIDEO_FPS,
                                frame_size)
        
        # Initialize processing variables
        window_size = Config.FRAMES_PER_PERIOD  # Process one period at a time
        window_start = 0
        tracked_objects = None  # Will maintain tracking information between windows
        last_grid_frame = None
        last_angle_grid_frame = None
        frames_per_period = 10
        
        print(f"\nInitializing video processing with:")
        print(f"Total frames: {frame_count}")
        print(f"Window size: {window_size} frames")
        
        while window_start < frame_count:
            window_end = min(window_start + window_size, frame_count)
            
            print(f"\nProcessing window: {window_start} to {window_end}")
            print(f"Using existing tracked objects: {len(tracked_objects) if tracked_objects else 0}")
            
            # Initialize window buffers for current period
            window_frames = []
            window_visualizations = []
            window_tracked_data = []
            
            # Seek to window start
            cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)
            frame_number = window_start
            
            print("Collecting detection data...")
            # Process frames in current window
            while frame_number < window_end:
                ret, frame = cap.read()
                if not ret:
                    break
                
                window_frames.append(frame.copy())
                
                self.frame_buffer.append(frame)
                roi_frame = frame[self.roi[1]:self.roi[1]+self.roi[3],
                                self.roi[0]:self.roi[0]+self.roi[2]]
                
                preprocessed = self._preprocess_frame(roi_frame, tracked_objects)
                current_frame_objects = self._detect_and_classify_objects(preprocessed, self.roi)
                self.detection_buffer.append(current_frame_objects)
                
                if len(self.detection_buffer) == 3:
                    averaged_objects = self._average_detections(list(self.detection_buffer))
                    tracked_objects = self._match_and_update_tracks(averaged_objects, tracked_objects)
                    
                    # Create visualization frame
                    vis_frame = self._create_visualization(frame, tracked_objects)
                    window_visualizations.append(vis_frame)
                    
                    # Store tracking data
                    if tracked_objects:
                        frame_data = {
                            'frame_number': frame_number,
                            'objects': [(obj.object_id, obj.object_type, obj.positions[-1], 
                                    obj.areas[-1], obj.contours[-1]) for obj in tracked_objects]
                        }
                        window_tracked_data.append(frame_data)
                    
                    self._record_detection(frame_number, tracked_objects)
                
                frame_number += 1
            
            print(f"Writing output for frames {window_start} to {window_end}...")
            # Write output for current window
            for idx, frame_number in enumerate(range(window_start, window_end)):
                if idx < len(window_visualizations):
                    vis_frame = window_visualizations[idx]
                    
                    # Write tracked video frames
                    for _ in range(5):  # Slow motion factor
                        out.write(vis_frame)
                    
                    # Create and write trajectory visualization
                    trajectory_frame = self._create_trajectory_visualization(
                        frame_size, frame_number, self.historical_tracks)
                    for _ in range(5):
                        trajectory_out.write(trajectory_frame)
                    
                    # Create and handle grid visualization
                    grid_frame = self._create_grid_visualization(
                        frame_size, frame_number, self.historical_tracks)
                    if grid_frame is not None:
                        last_grid_frame = grid_frame
                        grid_out.write(grid_frame)
                    elif last_grid_frame is not None:
                        grid_out.write(last_grid_frame)
            
                    # Create and handle angle grid visualization       
                    angle_grid_frame = self._create_angle_grid_visualization(
                        frame_size, frame_number, self.historical_tracks)
                    if angle_grid_frame is not None:
                        last_angle_grid_frame = angle_grid_frame
                        angle_grid_out.write(angle_grid_frame)
                    elif last_angle_grid_frame is not None:
                        angle_grid_out.write(last_angle_grid_frame)
            
            # Clear window buffers to free memory, but keep tracking information
            print("Clearing window buffers...")
            window_frames.clear()
            window_visualizations.clear()
            window_tracked_data.clear()
            
            # Move window forward
            window_start = window_end
            print(f"Moving window to frame {window_start}")
            print(f"Retained tracked objects: {len(tracked_objects) if tracked_objects else 0}")
        
        # Cleanup
        print("\nFinalizing processing...")
        cap.release()
        out.release()
        trajectory_out.release()
        grid_out.release()
        
        
        self._save_tracking_statistics()
        self._log_period_statistics()
        self._analyze_track_periods()
        self._analyze_track_continuity()
        self._create_aggregate_histogram() 
        
        print(f"Processing complete. Output saved to: {self.output_dir}")
        return output_path
    
def main():
    """Main function to run the cell tracking system."""
    video_path = "../../data/T2_Video26.mp4"  # Update with your video path
    
    try:
        tracker = CellTracker(video_path)
        output_path = tracker.process_video()
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()