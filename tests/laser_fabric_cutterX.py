import json
import logging
import math
import re
import threading
import traceback
import cv2
import numpy as np
import time
import os
import queue
import sys
from scipy.interpolate import splprep, splev




#sys.path.append('/home/yordam/balor') 
sys.path.append('../balor')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'balor')))
sys.path.append('/home/yordam/balor')
try:
    from balor.sender import Sender # type: ignore 
    from balor.command_list import CommandList # type: ignore
    print('Balor object loaded...')
except Exception as e:
    print(f'Error loading Balor object: {e}')
    print(f'Current sys.path: {sys.path}')
    print(f'Current working directory: {os.getcwd()}')
    sys.exit(1)


is_jetson_nano = os.path.exists('/dev/gpiochip0')
if is_jetson_nano:
    try:
        import Jetson.GPIO as GPIO
        print('Jetson.GPIO loaded...')
    except Exception as e:
        print(f'Error loading Jetson.GPIO: {e}')


has_encoder = True if is_jetson_nano else False
has_galvo = True if is_jetson_nano else False


# check if debug mode 
DEBUG = os.path.exists('data/debug')
print(f'DEBUG: {DEBUG}')

if is_jetson_nano:
    try:
        # load from tools folder
        from tools.encoder_speed import EncoderSpeed # type: ignore
        print(f'Loaded Encoder')
    except Exception as e:
        # load from local
        print(f'Loading error, {e}')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LaserFabricCutter:
    def __init__(self, video_path, settings, settings_file, save_frames=False, slice_start=320, slice_height=15, is_standalone=False):
        self.calibration_file = "data/calibration_file.json"
        
        self.y_hex = 44378  # Default value
        self.sender = None
        
        self.load_calibration()
        self.init_encoder()
        
        self.settings_file = settings_file
        self.video_path = video_path
        self.galvo_settings = settings
        self.output_folder = "processed_images"
        
        self.slice_start = slice_start
        self.slice_height = slice_height
        self.slice_end = self.slice_start + self.slice_height
        self.is_standalone = is_standalone
        self.save_frames = save_frames
        
        self.is_cutting = False
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.current_x = None     
        self.adjusted_y = None   
        self.cx_cm = 0  # 
   
        
        self.total_hex_distance = 51391 
        self.total_cm_distance = 16.3
        self.cm_per_hex_step = self.total_cm_distance / self.total_hex_distance
        self.hex_steps_per_cm = self.total_hex_distance / self.total_cm_distance
             
        self.sender = None
        self.galvo_connection = False
        self.galvo_control_thread = threading.Thread(target=self.connect_galvo_control)
        self.galvo_control_thread.daemon = True
        self.galvo_control_thread.start()
        self.fixed_y = slice_start
        
        #galvo points to queue
        self.is_running = False
        self.point_queue = queue.Queue(maxsize=100)   
          
        self.speed = 5  # Default speed, adjust as needed
        self.fabric_speed = 0.0
        self.last_point_time = None  # Son tespit edilen nokta zamanı
                       
        self.current_frame = None
        self.process_thread = None
        self.cutting_enabled = False

        self.threshold_percent = 15  
        self.initial_point_count = 10  
        self.initial_point_established = False  #
        self.initial_pattern_validated = False  
        self.is_paused = True 
        self.is_validation_mode = False
        self.validation_frame = None
        
        # Initialize the cutter
        self.galvo_thread = threading.Thread(target=self.galvo_cutting_loop)
        self.galvo_thread.daemon = True 
                        
        self.width = 0  # Add this line to store the width     
        self.calibration_mode = False
        
        self.last_valid_x = None
        self.last_valid_y = None

        self.galvo_loop_time = 0
        self.point_loop_time = 0
        self.use_point_deviation = False
        
        self.point_x = 0
        self.point_y = 0
        self.adjusted_x = 0
        self.adjusted_y = 0
        self.current_x_hex = 0
        self.current_y_hex = 0
        
        self.previous_points = set() 
                   
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
    
    def init_encoder(self):
        self.encoder = None
        if is_jetson_nano:
            try:
                self.encoder = EncoderSpeed()
            except Exception as e:
                print(f'Error loading Encoder: {e}')
                
    def load_calibration(self):
        try:
            with open(self.calibration_file, 'r') as file:
                calibration_data = json.load(file)
                self.pixel_cm_ratio = calibration_data.get('pixel_cm_ratio', 39.633)
                self.galvo_offset_x = calibration_data.get('galvo_offset_x', -35)
                self.galvo_offset_y = calibration_data.get('galvo_offset_y', 532)
                
            print(f'Loaded calibration: pixel_cm_ratio = {self.pixel_cm_ratio}, y_hex = {self.y_hex}, '
                  f'galvo_offset_x = {self.galvo_offset_x}, galvo_offset_y = {self.galvo_offset_y}')
        except FileNotFoundError:
            print(f"Calibration file {self.calibration_file} not found. Using default values.")
            self.galvo_offset_x = 40
            self.galvo_offset_y = 380
            self.pixel_cm_ratio = 39.633
            self.y_hex = 44378
    
    def save_calibration(self):
        calibration_data = {
            "pixel_cm_ratio": self.pixel_cm_ratio,
            "y_hex": self.y_hex,
            "galvo_offset_x": self.galvo_offset_x,
            "galvo_offset_y": self.galvo_offset_y
        }
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as file:
            json.dump(calibration_data, file)
        print(f"Calibration saved: galvo_offset_x = {self.galvo_offset_x}, galvo_offset_y = {self.galvo_offset_y}")
            
                
    def save_settings(self):
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        with open(self.settings_file, 'w') as f:
            json.dump(self.galvo_settings, f, indent=4)
                        
    ## GALVO SETTING            
    def update_galvo_settings(self, setting_name, value):
        print(f"Updating {setting_name} to {value}")
        self.galvo_settings[setting_name] = value
        if setting_name == 'slice_size':
            self.slice_end = self.slice_start + int(value)
            self.slice_height = int(value)
            
        threading.Thread(target=self.save_settings()).start()   
               
    # END galvo setting        
    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
        if not self.calibration_mode:
            # Exiting calibration mode, save the current offsets
            self.save_calibration()
    
    def toggle_pause(self):
        """Toggle the pause state of the video processing"""
        self.is_paused = not self.is_paused
        return self.is_paused
        
    def set_pause_state(self, state):
        """Explicitly set the pause state"""
        self.is_paused = state
        return self.is_paused

    def adjust_galvo_offset(self, dx, dy):
        with self.lock:
            self.galvo_offset_x += dx
            self.galvo_offset_y += dy  
            print(f"Galvo offset: X={self.galvo_offset_x}, Y={self.galvo_offset_y}")
                
    def calibrate_cm_pixel_ratio(self):
        # load calibration file due to version issue
        frame = self.get_current_frame()
        try:
            parameters = cv2.aruco.DetectorParameters_create()
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        except:
            parameters = cv2.aruco.DetectorParameters()
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


        try:
            # Create an ArucoDetector object and detect markers
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, _, _ = detector.detectMarkers(frame)
            
            if corners:
                aruco_perimeter = cv2.arcLength(corners[0], True)
                self.pixel_cm_ratio = aruco_perimeter / 20
                data = {
                    "pixel_cm_ratio": self.pixel_cm_ratio,
                    "y_hex": self.y_hex,
                    "galvo_offset_x": self.galvo_offset_x,
                    "galvo_offset_y": self.galvo_offset_y
                }
                with open(self.calibration_file, "w") as file:
                    json.dump(data, file)
                logging.info("Calibration successful!")
            else:
                print("No markers found!")
        except Exception as e:
            print("Aruco not found!", e)
        
    def connect_galvo_control(self):
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts and not self.galvo_connection:
            try:                
                self.sender = Sender()
                cor_table_data = open("tools/jetsonCalibrationdeneme1.cor", 'rb').read()
                if hasattr(self.sender, 'set_cor_table'):
                    self.sender.set_cor_table(cor_table_data)
                elif hasattr(self.sender, 'cor_table'):
                    self.sender.cor_table = cor_table_data
                self.sender.open(mock=DEBUG)
                self.galvo_connection = True
                self.galvo_zero_point()
                self.is_running = True
                
                if not self.galvo_thread.is_alive():
                    self.galvo_thread.start()
                    
            except Exception as e:
                attempt += 1
                logging.warning(f"Failed to connect to galvo (attempt {attempt}/{max_attempts}): {e}")
                self.galvo_connection = False
                self.sender = None
                time.sleep(2)
                
    def clamp_galvo_coordinates(self, x, y):
        # Assuming the valid range is 0-65535 (16-bit)
        x = max(0, min(x, 65535))
        y = max(0, min(y, 65535))
        return x, y    
                                                
    def pixel_to_galvo_coordinates(self, x, y):
        # Convert pixel coordinates to cm using the fixed ratio
        cm_x = x / self.pixel_cm_ratio
        cm_y = y / self.pixel_cm_ratio

        # Convert cm to hex steps
        hex_x = round(cm_x * self.hex_steps_per_cm)
        hex_y = round(cm_y * self.hex_steps_per_cm)

        # Clamp the coordinates to valid range
        hex_x, hex_y = self.clamp_galvo_coordinates(hex_x, hex_y)

        # Convert to hexadecimal and ensure 4-digit representation
        hex_x_str = f"{hex_x:04X}"
        hex_y_str = f"{hex_y:04X}"

        return hex_x, hex_y
                    
    def galvo_zero_point(self): # sıfır noktasına git
        self.sender.set_xy(0x0000, 0xC8FF)
        print ('galvo zero point')
    
    def calculate_loop_time(self):
        current_time = time.monotonic()
        if self.last_point_time is not None:
            self.point_loop_time = (current_time - self.last_point_time) 
        self.last_point_time = current_time  
        
    def get_current_frame(self):
        if self.current_frame is None:
            # Return a blank image if no frame is available
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.current_frame
        
    def validate_detection(self):
        """Smart validation that handles excess or missing groups"""
        self.is_validation_mode = True
        validation_frame = self.get_current_frame().copy()
        
        # Detect all potential groups
        detected_groups = self.detect_and_group_points(validation_frame)
        
        # Skip if no groups at all
        if not detected_groups:
            self.display_validation_result(validation_frame, [], False, "No groups detected")
            return False
        
        # Sort groups by x-position for consistent ordering
        sorted_groups = sorted(detected_groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
        
        # Case 1: Exact match - we have 4 groups
        if len(sorted_groups) == 4:
            self.reference_groups = sorted_groups
            self.display_validation_result(validation_frame, sorted_groups, True, "PASS - 4 groups detected")
            self.initial_pattern_validated = True
            return True
        
        # Case 2: Too many groups - select the best 4
        elif len(sorted_groups) > 4:
            # Filter groups based on spacing consistency and size
            filtered_groups = self.filter_best_groups(sorted_groups)
            
            self.reference_groups = filtered_groups
            self.display_validation_result(validation_frame, filtered_groups, True, 
                                        f"PASS - Selected best 4 from {len(sorted_groups)} groups")
            self.initial_pattern_validated = True
            return True
        
        # Case 3: Too few groups - try to infer missing ones
        elif 2 <= len(sorted_groups) < 4:
            # Try to infer the missing groups based on pattern
            complete_groups = self.infer_missing_groups(sorted_groups)
            
            if len(complete_groups) == 4:
                self.reference_groups = complete_groups
                self.display_validation_result(validation_frame, complete_groups, True, 
                                            f"PASS - Inferred {4-len(sorted_groups)} missing groups")
                self.initial_pattern_validated = True
                return True
            else:
                self.display_validation_result(validation_frame, sorted_groups, False, 
                                            f"FAIL - Could not infer missing groups. Need 4, found {len(sorted_groups)}")
                self.initial_pattern_validated = False
                return False
        
        # Case 4: Too few groups (less than 2) - can't reasonably infer
        else:
            self.display_validation_result(validation_frame, sorted_groups, False, 
                                        f"FAIL - Not enough groups. Need 4, found {len(sorted_groups)}")
            self.initial_pattern_validated = False
            return False

    def filter_best_groups(self, groups):
        """Filter out unlikely groups to get the best 4 groups"""
        if len(groups) <= 4:
            return groups
        
        # Sort by x-position
        sorted_groups = sorted(groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
        
        # Calculate average horizontal spacing between adjacent groups
        group_centers = [sum(p[0] for p in g)/len(g) for g in sorted_groups]
        spacings = [group_centers[i+1] - group_centers[i] for i in range(len(group_centers)-1)]
        
        if not spacings:
            return sorted_groups[:4]  # Fallback if spacing can't be calculated
        
        avg_spacing = sum(spacings) / len(spacings)
        
        # Score each group based on:
        # 1. Size (number of points) - prefer larger groups
        # 2. Spacing consistency - prefer groups that maintain consistent spacing
        group_scores = []
        
        for i, group in enumerate(sorted_groups):
            # Score based on size - larger is better
            size_score = len(group) / max(len(g) for g in sorted_groups)
            
            # Score based on spacing consistency
            spacing_score = 1.0
            if 0 < i < len(sorted_groups) - 1:
                # For middle groups, check spacing on both sides
                left_spacing = group_centers[i] - group_centers[i-1]
                right_spacing = group_centers[i+1] - group_centers[i]
                spacing_diff = abs(left_spacing - right_spacing) / avg_spacing
                spacing_score = 1.0 / (1.0 + spacing_diff)  # Higher score for more consistent spacing
            
            # Combined score (weight can be adjusted)
            total_score = (size_score * 0.6) + (spacing_score * 0.4)
            group_scores.append((i, total_score))
        
        # Sort by score and select top 4 indices
        top_indices = [idx for idx, _ in sorted(group_scores, key=lambda x: x[1], reverse=True)[:4]]
        top_indices.sort()  # Sort indices to maintain left-to-right order
        
        # Return the 4 best groups in proper order
        return [sorted_groups[i] for i in top_indices]

    def infer_missing_groups(self, groups):
        """Infer missing groups based on spacing pattern"""
        if len(groups) >= 4:
            return groups
        
        # Sort by x-position
        sorted_groups = sorted(groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
        
        # Get centers of known groups
        group_centers = [sum(p[0] for p in g)/len(g) for g in sorted_groups]
        
        # If we have at least 2 groups, we can estimate spacing
        if len(sorted_groups) >= 2:
            # Calculate available spacings
            spacings = [group_centers[i+1] - group_centers[i] for i in range(len(group_centers)-1)]
            avg_spacing = sum(spacings) / len(spacings)
            
            # Create template for missing groups
            template_group = None
            avg_points = sum(len(g) for g in sorted_groups) / len(sorted_groups)
            
            # Use the group with points count closest to average as template
            min_diff = float('inf')
            for group in sorted_groups:
                diff = abs(len(group) - avg_points)
                if diff < min_diff:
                    min_diff = diff
                    template_group = group
            
            # Get y-coordinates from template
            template_y = [p[1] for p in template_group]
            
            # Complete the pattern to 4 groups
            complete_groups = sorted_groups.copy()
            frame_width = 640  # Assuming standard width, adjust if needed
            
            # Case 1: Missing a group on the left
            if len(group_centers) >= 2 and (group_centers[0] - avg_spacing > 0):
                # Infer a group to the left of the first group
                inferred_center = group_centers[0] - avg_spacing
                inferred_group = [(inferred_center, y) for y in template_y]
                complete_groups.append(inferred_group)
            
            # Case 2: Missing a group on the right
            if len(group_centers) >= 2 and (group_centers[-1] + avg_spacing < frame_width):
                # Infer a group to the right of the last group
                inferred_center = group_centers[-1] + avg_spacing
                inferred_group = [(inferred_center, y) for y in template_y]
                complete_groups.append(inferred_group)
            
            # Case 3: Missing a group in the middle
            if len(group_centers) >= 2:
                for i in range(len(group_centers) - 1):
                    current_spacing = group_centers[i+1] - group_centers[i]
                    # If space between centers is significantly larger than average, infer a group
                    if current_spacing > 1.5 * avg_spacing:
                        inferred_center = group_centers[i] + avg_spacing
                        inferred_group = [(inferred_center, y) for y in template_y]
                        complete_groups.append(inferred_group)
            
            # Re-sort the groups
            complete_groups = sorted(complete_groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            
            # If we still don't have 4 groups, try one more time with updated spacing
            if len(complete_groups) < 4:
                # Recalculate centers and spacing
                group_centers = [sum(p[0] for p in g)/len(g) for g in complete_groups]
                spacings = [group_centers[i+1] - group_centers[i] for i in range(len(group_centers)-1)]
                if spacings:
                    avg_spacing = sum(spacings) / len(spacings)
                    
                    # Try to infer groups at both ends if needed
                    while len(complete_groups) < 4:
                        # Decide whether to add to left or right based on frame boundaries
                        if group_centers[0] - avg_spacing > 0:
                            # Add to left
                            inferred_center = group_centers[0] - avg_spacing
                            inferred_group = [(inferred_center, y) for y in template_y]
                            complete_groups.append(inferred_group)
                        elif group_centers[-1] + avg_spacing < frame_width:
                            # Add to right
                            inferred_center = group_centers[-1] + avg_spacing
                            inferred_group = [(inferred_center, y) for y in template_y]
                            complete_groups.append(inferred_group)
                        else:
                            # Can't add more groups within frame
                            break
                        
                        # Update centers
                        complete_groups = sorted(complete_groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
                        group_centers = [sum(p[0] for p in g)/len(g) for g in complete_groups]
        
        # Return what we have (may or may not be 4 groups)
        return sorted(complete_groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)[:4]

    def display_validation_result(self, frame, groups, is_valid, message):
        """Display validation results on the frame"""
        for i, group in enumerate(groups):
            # Use different colors for each group
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            color = colors[i % len(colors)]
            
            # Draw points and add group labels
            for x, y in group:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            
            # Add group label
            if group:
                avg_x = sum(p[0] for p in group) / len(group)
                avg_y = sum(p[1] for p in group) / len(group)
                
                # For inferred groups, add "(inferred)" to the label
                is_inferred = i >= len(groups) - (4 - len(set(tuple(p) for g in groups for p in g)))
                label = f"Group {i+1}"
                if is_inferred:
                    label += " (inferred)"
                    
                cv2.putText(frame, label, (int(avg_x), int(avg_y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add message at the top
        cv2.putText(frame, message, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 0, 255) if not is_valid else (0, 255, 0), 2)
        
        # Push the visualization frame to the queue
        if hasattr(self, 'frame_queue') and self.frame_queue is not None:
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame.copy())
            except Exception as e:
                print(f"Error updating frame queue: {e}")
                
        # Store validation frame
        self.validation_frame = frame.copy()
      
    def detect_and_group_points(self, frame):
        """Process frame to detect and group points by x-coordinate proximity"""
        offset_px = 25
        slice_img = frame[self.slice_start:self.slice_end, :]
        
        # Apply existing image processing steps
        gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter valid contours
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if area > 500 and h > self.slice_height*0.8:
                valid_contours.append(cnt)
        
        # Extract centerline points from each contour
        all_points = []
        for cnt in valid_contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            for y in range(self.slice_height):
                row = mask[y,:]
                if np.any(row):
                    left = np.argmax(row)
                    right = len(row) - np.argmax(row[::-1]) - 1
                    cx = (left + right) // 2
                    all_points.append((cx - offset_px, y + self.slice_start))
                    all_points.append((cx + offset_px, y + self.slice_start))
        
        # Group points by x-coordinate proximity
        if not all_points:
            return []
            
        # Sort points by x-coordinate
        all_points.sort(key=lambda p: p[0])
        
        # Use a clustering approach with a larger gap threshold
        x_values = [p[0] for p in all_points]
        
        # Find large gaps to separate the groups
        gaps = [x_values[i+1] - x_values[i] for i in range(len(x_values)-1)]
        if not gaps:
            return [all_points]  # Only one group if no gaps
            
        # Find the 3 largest gaps to separate 4 groups
        if len(gaps) >= 3:
            # Get indices of the 3 largest gaps
            gap_indices = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)[:3]
            gap_indices.sort()  # Sort in ascending order
            
            # Create groups based on these gap positions
            groups = []
            start_idx = 0
            
            for gap_idx in gap_indices:
                groups.append(all_points[start_idx:gap_idx+1])
                start_idx = gap_idx + 1
                
            # Add the final group
            groups.append(all_points[start_idx:])
        else:
            # Fallback if we don't have enough gaps
            groups = [all_points]
        
        return groups
    
    def exit_validation_mode(self):
        """Exit validation mode but maintain continuous detection"""
        self.is_validation_mode = False
        
        # Clear any validation-specific visualization
        current_frame = self.get_current_frame().copy()
        
        # Reset the validation frame to ensure it doesn't persist
        self.validation_frame = None
        
        # Ensure detection will continue on subsequent frames
        if hasattr(self, 'frame_queue') and self.frame_queue is not None:
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                # Put a clean frame to reset the display
                self.frame_queue.put(current_frame)
            except:
                pass

                    
    def process_video(self, frame_queue=None):
        print("Starting video processing...")
        if frame_queue is None:
            frame_queue = queue.Queue(maxsize=1)
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Handle different input types
            if isinstance(self.video_path, dict) and self.video_path["type"] == "image":
                # Static image mode
                frame = cv2.imread(self.video_path["path"])
                if frame is None:
                    print(f"Error: Could not load image: {self.video_path['path']}")
                    return
                    
                while self.is_running:
                    self.process_frame(frame.copy(), frame_queue)
                    time.sleep(0.01)  # Small delay
                    
            else:
                # Video or camera mode
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    print(f"Error: Could not open video source: {self.video_path}")
                    return
                    
                # Get the original video's frame rate
                original_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if original_fps <= 0 or np.isnan(original_fps):
                    original_fps = 30.0  # Default to 30fps if unable to determine
                # Calculate the delay needed between frames to maintain original speed
                frame_delay = 1.0 / original_fps
                
                # For pause functionality: store the current frame for display during pause
                ret, current_frame = self.cap.read()
                if not ret:
                    print("Failed to read initial frame")
                    return
                    
                # Disable autofocus for Logitech C920
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 disables autofocus
                self.cap.set(cv2.CAP_PROP_FOCUS, 50)     # Set a fixed focus value (0-255, adjust as needed)
                # Camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                prev_frame_time = time.time()
                
                while self.cap.isOpened() and self.is_running:
                            # Check if in validation mode - if so, keep showing validation frame
                    if self.is_validation_mode and self.validation_frame is not None:
                        # Just keep showing the validation frame
                        self.update_frame_queue(self.validation_frame.copy(), frame_queue)
                        time.sleep(0.03)  # Small delay to avoid CPU overuse
                        continue
        
                    if not self.is_paused:
                        # Only read a new frame if not paused
                        ret, current_frame = self.cap.read()
                        if not ret or current_frame is None:
                            if isinstance(self.video_path, str):  # Video file
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                                continue
                            else:  # Camera error
                                print("Frame capture failed")
                                break
                    
                    # Always process the current frame (either new or same as before if paused)
                    self.process_frame(current_frame.copy(), frame_queue)               
                    
                    # Control frame rate to match original video
                    current_time = time.time()
                    elapsed = current_time - prev_frame_time
                    
                    # Only sleep if we're not paused
                    if not self.is_paused:
                        sleep_time = max(0, frame_delay - elapsed)
                        time.sleep(sleep_time)
                    else:
                        # When paused, just add a small delay to prevent CPU overuse
                        time.sleep(0.03)
                        
                    prev_frame_time = time.time() 
                    
        except Exception as e:
            print("Error in image processing:", e)
            traceback.print_exc()
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

    # TODO
    def process_frame(self, frame, frame_queue):
        """Process frame with improved detection and continuity checks"""
        # Get current groups of points
        current_groups = self.detect_and_group_points(frame)
        
        # Skip if no groups detected
        if not current_groups:
            self.update_frame_queue(frame, frame_queue)
            return
        
        # Always visualize the detected groups, even if not cutting
        for i, group in enumerate(current_groups):
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            color = colors[i % len(colors)]
            for x, y in group:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        
        # If initial pattern not validated and we're not in cutting mode, just show detection
        if not self.initial_pattern_validated and not self.is_cutting:
            # Draw detection for visualization            
            self.update_frame_queue(frame, frame_queue)
            return
        
        # Check continuity with previous valid groups when cutting is active
        valid_groups = []
        
        if self.is_cutting and hasattr(self, 'previous_valid_groups') and self.previous_valid_groups:
            # Calculate movement/deviation from previous frame
            deviation_threshold = self.galvo_settings['point_daviation']
            
            for group in current_groups:
                # Find closest previous group
                group_center_x = sum(p[0] for p in group) / len(group)
                
                # Find the closest previous group by comparing centers
                prev_centers = [sum(p[0] for p in g) / len(g) for g in self.previous_valid_groups]
                closest_idx = min(range(len(prev_centers)), 
                                key=lambda i: abs(prev_centers[i] - group_center_x))
                
                # Calculate average deviation
                avg_deviation = 0
                if self.previous_valid_groups[closest_idx]:
                    deviations = []
                    for p1 in group:
                        closest_point = min(self.previous_valid_groups[closest_idx], 
                                        key=lambda p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                        deviations.append(math.sqrt((p1[0]-closest_point[0])**2 + 
                                                (p1[1]-closest_point[1])**2))
                    
                    avg_deviation = sum(deviations) / len(deviations)
                
                # Accept the group if deviation is within threshold
                if avg_deviation <= deviation_threshold or avg_deviation == 0:
                    valid_groups.append(group)
        else:
            # No previous groups to compare with, accept all
            valid_groups = current_groups
        
        # Update previous groups for next frame
        if valid_groups:
            self.previous_valid_groups = valid_groups.copy()
        
        # Process valid groups for galvo
        if valid_groups and self.is_cutting:
            # Clear previous points
            self.previous_points.clear()
            
            # Process each group in sequence
            for group_idx, group in enumerate(valid_groups):
                # Convert to galvo coordinates
                galvo_points = []
                for x, y in group:
                    adjusted_x = x + self.galvo_offset_x
                    adjusted_y = y + self.galvo_offset_y + self.galvo_settings['offset_y']
                    
                    # Convert to hex coordinates
                    x_hex, y_hex = self.pixel_to_galvo_coordinates(adjusted_x, adjusted_y)
                    galvo_points.append((x_hex, y_hex))
                
                # Add points to processing queue with group metadata
                self.process_point_group(galvo_points, group_idx)
                
                # Draw the processed points
                color = (0, 0, 255)  # Red for valid cutting groups
                for x, y in group:
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        
        # Update the frame queue
        self.calculate_loop_time()
        self.update_frame_queue(frame, frame_queue)
        
    def process_point_group(self, points, group_index):
        """Process a group of points, optimizing galvo movement"""
        if not points:
            return
            
        # If this is a new group, first move to its starting point without cutting
        first_point = points[0]
        
        # Queue the sequence: first move without cutting, then cut the entire group
        if self.galvo_connection:
            # First just move to the group's starting position
            self.point_queue.put((first_point[0], first_point[1], False))  # False = no cutting
            
            # Then put all points with cutting enabled
            for point in points:
                self.point_queue.put((point[0], point[1], True))  # True = cutting

    def is_valid_point(self, x, y):
        if not self.use_point_deviation:
            return True
        
        # Referance deviation point will be fixed until cutting is started.
        if self.last_valid_x is None or self.last_valid_y is None or not self.is_cutting:
            # First point is always valid
            self.last_valid_x, self.last_valid_y = x, y
            return True
        
        # Calculate distance from last valid point
        distance = math.sqrt((x - self.last_valid_x)**2 + (y - self.last_valid_y)**2)

        if distance <=  self.galvo_settings['point_daviation']:
            return True
        else:
            print(f"Invalid point: {x}, {y}. Distance: {distance}")
            return False
        
    def send_point_to_galvo(self, x_hex, y_hex):
        if not self.galvo_connection:
            #logging.info(f"Galvo not connected. Skipping point. Dummy galvo: Sending point ({x_hex}, {y_hex})")
            return
        try:
            with self.lock:
                self.point_queue.put_nowait((x_hex, y_hex))
            self.current_x_hex = x_hex
            self.current_y_hex = y_hex
        except queue.Full:
            logging.warning("Point queue is full. Skipping point.")
        except Exception as e:
            logging.error(f"Error sending point to galvo: {e}")
                                
    def update_frame_queue(self, frame, frame_queue):
        try:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass  # Queue is already empty, no need to remove
            frame_queue.put_nowait(frame)
            self.current_frame = frame.copy()
        except queue.Full:
            pass  # Queue is full, skip this frame
        
    def start_cutting(self):
        self.is_cutting = True
        if is_jetson_nano and self.encoder:
            self.encoder.start_cutting()
                    
        for setting_name, value in self.galvo_settings.items():
            self.update_galvo_settings(setting_name, value)
            
    def stop_cutting(self):
        self.is_cutting = False
        
        if is_jetson_nano and self.encoder:
            self.encoder.stop_cutting()
        
        self.return_to_start_position()


            
    def start_processing(self, frame_queue):
        print("Starting processing thread")
        with self.lock:
            self.is_running = True
        if self.process_thread is None or not self.process_thread.is_alive():
            self.process_thread = threading.Thread(target=self.process_video, args=(frame_queue,))
            self.process_thread.start()
            
        if not self.galvo_thread.is_alive():
            self.galvo_thread.start()
    
    def stop_processing(self):
        self.return_to_start_position()
        with self.lock:
            self.is_cutting = False
            self.is_running = False
        if self.process_thread:
            self.process_thread.join()
                               
    def get_all_points(self):
        points = []
        with self.lock:
            while not self.point_queue.empty():
                points.append(self.point_queue.get())
        return points
                 
    # TODO: galvo_cutting_loop
    def galvo_cutting_loop(self):
        if not self.sender:
            logging.error("Sender is not initialized")
            return

        # Include relevant keys for laser settings
        include_keys = [
            'travel_speed', 'frequency', 'power', 'cut_speed',
            'laser_on_delay', 'laser_off_delay', 'polygon_delay'
        ]

        while self.is_running and not self.stop_event.is_set():
            # Gather current laser settings
            params = {k: v for k, v in self.galvo_settings.items() if k in include_keys}
            start_time = time.monotonic()

            try:
                # Get all points from the queue (now with cutting flag)
                points_batch = []
                while not self.point_queue.empty() and len(points_batch) < 100:  # Limit batch size
                    try:
                        point = self.point_queue.get_nowait()
                        points_batch.append(point)
                    except queue.Empty:
                        break

                if points_batch:
                    # Separate movement points from cutting points
                    move_points = [p[:2] for p in points_batch if len(p) > 2 and not p[2]]
                    cut_points = [p[:2] for p in points_batch if len(p) > 2 and p[2]]
                    
                    # Process movement points first
                    if move_points:
                        # Just move to these points without cutting
                        for point in move_points:
                            self.sender.set_xy(point[0], point[1])
                    
                    # Then process cutting points
                    if cut_points and self.is_cutting:
                        def tick(cmds, loop_index):
                            cmds.clear()
                            cmds.set_mark_settings(**params)
                            
                            for point in cut_points:
                                cmds.mark(point[0], point[1])
                        
                        job = self.sender.job(tick=tick)
                        job.execute(1)

            except Exception as e:
                logging.error(f"Error in galvo_loop: {e}")

            # Manage loop timing
            galvo_loop_time = time.monotonic() - start_time
            time.sleep(max(0, self.point_loop_time - galvo_loop_time))
            self.galvo_loop_time = time.monotonic() - start_time


        # Ensure laser is turned off and moves to start when the loop ends
        if self.sender:
            try:
                self.sender.raw_disable_laser()  # Turn off laser
                self.sender.set_xy(0x8000, 0x8000)  # Move to neutral/start position
            except Exception as e:
                logging.error(f"Failed to disable laser: {e}")

                        
    def stop_sender_loop(self):           
        self.stop_event.set()
        if self.sender:
            self.sender.abort()
        if self.galvo_thread and self.galvo_thread.is_alive():
            self.galvo_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to finish
            
            
    def return_to_start_position(self):
        # Implement the logic to return to the start position
        if self.galvo_connection:
            self.galvo_zero_point()
        logging.info("Returned to start position")
                    
    def __del__(self):
        try:
            if self.sender and hasattr(self.sender, 'close'):
                self.sender.close()
        except Exception as e:
            print(f"Error during LaserFabricCutter cleanup: {e}")
        
    def cleanup(self):
        # Set flags first
        self.is_running = False
        self.is_cutting = False
        
        # Stop all threads
        self.stop_event.set()
        
        try:
            # Stop galvo operations with timeout
            if self.sender:
                stop_thread = threading.Thread(target=self._safe_stop_sender)
                stop_thread.daemon = True
                stop_thread.start()
                stop_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Galvo shutdown error: {e}")
        
        # Clean up resources
        if self.encoder:
            self.encoder.cleanup()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
    def _safe_stop_sender(self):
        try:
            if self.sender:
                self.sender.abort()
                self.sender.close()
        except Exception as e:
            print(f"Sender stop error: {e}")
                                
# Start laser cutter without GUI
if __name__ == "__main__":
    video_source = 0 # 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method=2 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    cutter = LaserFabricCutter(video_source, slice_height=5, is_standalone=True)
    time.sleep(.1)    
    try:
        cutter.is_running = True
        cutter.process_video()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        cutter.cleanup()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        cutter.cleanup()