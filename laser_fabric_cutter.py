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
        
        # Add new parameters for channel selection
        self.left_channel_start = self.galvo_settings['left_channel_start']
        self.left_channel_end = self.galvo_settings['left_channel_end']
        self.right_channel_start = self.galvo_settings['right_channel_start']
        self.right_channel_end = self.galvo_settings['right_channel_end']
        self.offset_px = self.galvo_settings['offset_px']
    
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

    def update_channel_boundaries(self, left_start, left_end, right_start, right_end):
        with self.lock:
            self.left_channel_start = left_start
            self.left_channel_end = left_end
            self.right_channel_start = right_start
            self.right_channel_end = right_end
            self.galvo_settings['left_channel_start'] = left_start
            self.galvo_settings['left_channel_end'] = left_end
            self.galvo_settings['right_channel_start'] = right_start
            self.galvo_settings['right_channel_end'] = right_end
            threading.Thread(target=self.save_settings()).start() 
            print(f"Updated channel boundaries: Left({left_start}-{left_end}), Right({right_start}-{right_end})")

    def update_offset_px(self, value):
        with self.lock:
            self.offset_px = value
            print(f"Updated offset_px: {value}")
                               
    def update_galvo_settings(self, setting_name, value):
        print(f"Updating {setting_name} to {value}")
        self.galvo_settings[setting_name] = value
        if setting_name == 'slice_size':
            self.slice_end = self.slice_start + int(value)
            self.slice_height = int(value)
            
        threading.Thread(target=self.save_settings()).start()   
                     
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
    
    def detect_and_group_points(self, frame):
        """Process frame to detect and group points by x-coordinate proximity within allowed channels"""
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
            
            # Check if contour is within the allowed channel boundaries
            contour_center_x = x + w/2
            is_in_left_channel = (contour_center_x >= self.left_channel_start and 
                                contour_center_x <= self.left_channel_end)
            is_in_right_channel = (contour_center_x >= self.right_channel_start and 
                                contour_center_x <= self.right_channel_end)
            
            if area > 500 and h > self.slice_height*0.8 and (is_in_left_channel or is_in_right_channel):
                valid_contours.append(cnt)
        
        # Extract centerline points from each contour
        left_channel_points = []
        right_channel_points = []
        
        for cnt in valid_contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Get contour center to determine which channel it belongs to
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
            else:
                x, _, w, _ = cv2.boundingRect(cnt)
                cx = x + w//2
                
            is_left_channel = (cx >= self.left_channel_start and cx <= self.left_channel_end)
            
            for y in range(self.slice_height):
                row = mask[y,:]
                if np.any(row):
                    left = np.argmax(row)
                    right = len(row) - np.argmax(row[::-1]) - 1
                    center_x = (left + right) // 2
                    
                    # Add points with offset to the appropriate channel
                    if is_left_channel:
                        # Only add if the points are within the left channel boundaries
                        if self.left_channel_start <= center_x - self.offset_px <= self.left_channel_end:
                            left_channel_points.append((center_x - self.offset_px, y + self.slice_start))
                        if self.left_channel_start <= center_x + self.offset_px <= self.left_channel_end:
                            left_channel_points.append((center_x + self.offset_px, y + self.slice_start))
                    else:
                        # Only add if the points are within the right channel boundaries
                        if self.right_channel_start <= center_x - self.offset_px <= self.right_channel_end:
                            right_channel_points.append((center_x - self.offset_px, y + self.slice_start))
                        if self.right_channel_start <= center_x + self.offset_px <= self.right_channel_end:
                            right_channel_points.append((center_x + self.offset_px, y + self.slice_start))
        
        # Process each channel separately to find exactly 2 groups per channel
        left_groups = self.find_two_groups_in_channel(left_channel_points)
        right_groups = self.find_two_groups_in_channel(right_channel_points)
        
        # Combine the groups from both channels
        all_groups = left_groups + right_groups
        
        # If we don't have enough groups, try to infer the missing ones
        if len(all_groups) < 4:
            all_groups = self.infer_missing_groups(all_groups)
        
        # Sort all groups by x-position
        all_groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
        
        return all_groups[:4]  # Return at most 4 groups

    def find_two_groups_in_channel(self, points):
        """Find exactly two groups of points within a single channel"""
        if not points:
            return []
            
        # If we have very few points, just return them as a single group
        if len(points) < 6:
            return [points]
        
        try:
            # Convert points to numpy array
            points_array = np.array(points)
            
            # Use only x-coordinates for clustering
            x_coords = points_array[:, 0].reshape(-1, 1)
            
            # Always try to find 2 clusters within each channel
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x_coords)
            
            # Get cluster labels
            labels = kmeans.labels_
            
            # Create groups based on cluster labels
            groups = [[] for _ in range(2)]
            for i, point in enumerate(points):
                groups[labels[i]].append(point)
            
            # Filter out empty groups
            groups = [g for g in groups if g]
            
            # Sort groups by x-coordinate
            groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            
            # Optimize each group
            optimized_groups = []
            for group in groups:
                optimized_group = self.optimize_point_group(group)
                if optimized_group:
                    optimized_groups.append(optimized_group)
                    
            return optimized_groups
            
        except Exception as e:
            print(f"Error finding groups in channel: {e}")
            
            # Fallback: sort by x-coordinate and split in the middle
            points.sort(key=lambda p: p[0])
            mid = len(points) // 2
            
            group1 = self.optimize_point_group(points[:mid])
            group2 = self.optimize_point_group(points[mid:])
            
            return [group1, group2] if group1 and group2 else [points]

    def cluster_points_into_four_groups(self, points):
        """
        Cluster points into exactly four groups using a combination of
        spatial clustering and domain knowledge about the expected pattern.
        """
        # Sort points by x-coordinate
        points.sort(key=lambda p: p[0])
        
        # Step 1: Initial clustering based on x-coordinate gaps
        x_values = [p[0] for p in points]
        
        # Calculate gaps between adjacent x-coordinates
        gaps = [x_values[i+1] - x_values[i] for i in range(len(x_values)-1)]
        if not gaps:
            return [points]  # Only one group if no gaps
        
        # Step 2: Use k-means clustering to find clusters
        try:
            # Convert points to numpy array for k-means
            points_array = np.array(points)
            
            # Use only x-coordinates for clustering
            x_coords = points_array[:, 0].reshape(-1, 1)
            
            # Determine the number of clusters to use
            # First, try to estimate the number of natural clusters
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Try different numbers of clusters (2-4) and pick the best
            best_score = -1
            best_k = 2
            for k in range(2, 5):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(x_coords)
                    if len(np.unique(kmeans.labels_)) > 1:  # Ensure we have at least 2 clusters
                        score = silhouette_score(x_coords, kmeans.labels_)
                        if score > best_score:
                            best_score = score
                            best_k = k
                except:
                    continue
            
            # Apply k-means with the best k
            kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10).fit(x_coords)
            
            # Get cluster labels
            labels = kmeans.labels_
            
            # Create groups based on cluster labels
            groups = [[] for _ in range(best_k)]
            for i, point in enumerate(points):
                groups[labels[i]].append(point)
            
            # Sort groups by x-coordinate (left to right)
            groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            
        except Exception as e:
            # Fallback method if k-means fails
            print(f"K-means clustering failed: {e}. Using fallback method.")
            
            # Find the largest gaps to separate groups
            if len(gaps) >= 1:
                # Sort gaps by size (largest first)
                gap_indices = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)
                
                # Use the number of large gaps to determine number of groups
                # (at most 3 gaps for 4 groups)
                num_gaps = min(3, len(gap_indices))
                gap_indices = sorted(gap_indices[:num_gaps])
                
                groups = []
                start_idx = 0
                for gap_idx in gap_indices:
                    groups.append(points[start_idx:gap_idx+1])
                    start_idx = gap_idx + 1
                groups.append(points[start_idx:])
            else:
                # Not enough gaps, just use one group
                groups = [points]
        
        # Step 3: Optimize each group by reducing redundant points
        optimized_groups = []
        for group in groups:
            if group:  # Only process non-empty groups
                optimized_group = self.optimize_point_group(group)
                if optimized_group:  # Only add non-empty groups
                    optimized_groups.append(optimized_group)
        
        # Step 4: Ensure we have exactly 4 groups
        while len(optimized_groups) > 4:
            # Merge the two closest groups
            min_distance = float('inf')
            merge_indices = (0, 1)
            
            for i in range(len(optimized_groups)):
                for j in range(i+1, len(optimized_groups)):
                    if optimized_groups[i] and optimized_groups[j]:  # Check for non-empty groups
                        g1_center = sum(p[0] for p in optimized_groups[i])/len(optimized_groups[i])
                        g2_center = sum(p[0] for p in optimized_groups[j])/len(optimized_groups[j])
                        distance = abs(g1_center - g2_center)
                        
                        if distance < min_distance:
                            min_distance = distance
                            merge_indices = (i, j)
            
            # Merge the two closest groups
            i, j = merge_indices
            optimized_groups[i].extend(optimized_groups[j])
            optimized_groups.pop(j)
        
        # If we have fewer than 4 groups, try to infer the missing ones
        while len(optimized_groups) < 4:
            new_groups = self.infer_missing_groups(optimized_groups)
            if len(new_groups) == len(optimized_groups):
                # If inference didn't add any new groups, break to avoid infinite loop
                break
            optimized_groups = new_groups
        
        # Final sort by x-coordinate
        optimized_groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
        
        # Ensure all groups have valid points
        for i in range(len(optimized_groups)):
            if not optimized_groups[i]:
                # Replace empty group with a dummy point that won't be drawn
                # This prevents errors when trying to draw circles
                optimized_groups[i] = [(-1000, -1000)]
        
        return optimized_groups[:4]  # Return at most 4 groups


    def infer_missing_groups(self, groups):
        """Infer missing groups based on spacing pattern"""
        if len(groups) >= 4:
            return groups
        
        # Initialize complete_groups at the beginning to avoid UnboundLocalError
        complete_groups = groups.copy()
        
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
            frame_width = 1280  # Assuming standard width, adjust if needed
            
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
            complete_groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            
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
                        complete_groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
                        group_centers = [sum(p[0] for p in g)/len(g) for g in complete_groups]
        else:
            # If we have fewer than 2 groups, we need a different approach
            if len(sorted_groups) == 1 and sorted_groups[0]:
                # We have only one group, try to infer the other three
                template_group = sorted_groups[0]
                template_y = [p[1] for p in template_group]
                
                # Get the center of the existing group
                center_x = sum(p[0] for p in template_group) / len(template_group)
                
                # Estimate spacing based on channel width
                estimated_spacing = (self.right_channel_start - self.left_channel_end) / 3
                
                # Create three more groups
                for i in range(1, 4):
                    # Alternate between left and right of the existing group
                    if i % 2 == 1:
                        # Add to right
                        new_center = center_x + (i * estimated_spacing)
                    else:
                        # Add to left
                        new_center = center_x - (i * estimated_spacing)
                    
                    # Ensure the new center is within frame
                    if 0 < new_center < 1280:
                        inferred_group = [(new_center, y) for y in template_y]
                        complete_groups.append(inferred_group)
                
                # Re-sort the groups
                complete_groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            else:
                # No valid groups to infer from, create dummy groups
                dummy_group = [(-1000, -1000)]  # Off-screen point
                while len(complete_groups) < 4:
                    complete_groups.append(dummy_group.copy())
        
        # Return what we have (may or may not be 4 groups)
        return sorted(complete_groups, key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)[:4]

    def optimize_point_group(self, points, max_points=10):
        """
        Optimize a group of points by:
        1. Fitting points to a straight line using linear regression
        2. Generating evenly spaced points along this line
        3. Limiting the total number of points for efficiency
        """
        if not points or len(points) < 2:
            return points
        
        # Sort points by y-coordinate (top to bottom)
        points.sort(key=lambda p: p[1])
        
        # If we have few points, still apply linear regression
        if len(points) <= 3:
            # Need at least 2 points for regression
            return self.linearize_points(points, max_points)
        
        # For larger groups, use linear regression to create a straight line
        return self.linearize_points(points, max_points)

    def linearize_points(self, points, num_points=10):
        """
        Convert a potentially scattered group of points into a straight line
        using linear regression, then generate evenly spaced points along this line.
        """
        if len(points) < 2:
            return points
        
        try:
            # Extract x and y coordinates
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # Check if points are more vertical or horizontal
            y_range = max(y_coords) - min(y_coords)
            x_range = max(x_coords) - min(x_coords)
            
            if y_range > x_range:
                # More vertical - fit y = mx + b
                # Reshape for sklearn
                x_for_fit = x_coords.reshape(-1, 1)
                
                # Use linear regression to find the best fit line
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(x_for_fit, y_coords)
                
                # Get the slope and intercept
                slope = model.coef_[0]
                intercept = model.intercept_
                
                # Generate evenly spaced y-coordinates
                min_y = min(y_coords)
                max_y = max(y_coords)
                new_y_coords = np.linspace(min_y, max_y, num_points)
                
                # Calculate corresponding x-coordinates using the line equation
                # x = (y - b) / m
                if slope != 0:
                    new_x_coords = (new_y_coords - intercept) / slope
                else:
                    # If slope is 0 (horizontal line), use the average x
                    new_x_coords = np.full_like(new_y_coords, np.mean(x_coords))
                
                # Create new points
                linearized_points = [(int(round(x)), int(round(y))) 
                                for x, y in zip(new_x_coords, new_y_coords)]
            else:
                # More horizontal - fit x = my + b (inverse regression)
                # Reshape for sklearn
                y_for_fit = y_coords.reshape(-1, 1)
                
                # Use linear regression to find the best fit line
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(y_for_fit, x_coords)
                
                # Get the slope and intercept
                slope = model.coef_[0]
                intercept = model.intercept_
                
                # Generate evenly spaced x-coordinates
                min_x = min(x_coords)
                max_x = max(x_coords)
                new_x_coords = np.linspace(min_x, max_x, num_points)
                
                # Calculate corresponding y-coordinates using the line equation
                # y = (x - b) / m
                if slope != 0:
                    new_y_coords = (new_x_coords - intercept) / slope
                else:
                    # If slope is 0 (vertical line), use the average y
                    new_y_coords = np.full_like(new_x_coords, np.mean(y_coords))
                
                # Create new points
                linearized_points = [(int(round(x)), int(round(y))) 
                                for x, y in zip(new_x_coords, new_y_coords)]
            
            return linearized_points
            
        except Exception as e:
            print(f"Linearization failed: {e}")
            # Fallback to simple sampling if regression fails
            return self.simple_sample_points(points, num_points)
        
    def simple_sample_points(self, points, num_points):
        """Simple fallback method to sample points evenly along y-axis"""
        if len(points) <= num_points:
            return points
        
        # Sort by y-coordinate
        points.sort(key=lambda p: p[1])
        
        # Always include the first and last points
        result = [points[0]]
        
        # Sample the middle points
        if num_points > 2:
            step = (len(points) - 2) / (num_points - 2)
            for i in range(1, num_points - 1):
                idx = min(int(i * step), len(points) - 2)
                result.append(points[idx])
        
        # Add the last point
        result.append(points[-1])
        
        return result


    # TODO 
    def process_frame(self, frame, frame_queue):
        """Process frame with improved detection and continuity checks"""
        # Get current groups of points
        current_groups = self.detect_and_group_points(frame)
        
        # Skip if no groups detected
        if not current_groups:
            self.update_frame_queue(frame, frame_queue)
            return
        
        # Draw channel boundaries
        cv2.line(frame, (self.left_channel_start, self.slice_start), 
                (self.left_channel_start, self.slice_end), (0, 255, 255), 2)
        cv2.line(frame, (self.left_channel_end, self.slice_start), 
                (self.left_channel_end, self.slice_end), (0, 255, 255), 2)
        cv2.line(frame, (self.right_channel_start, self.slice_start), 
                (self.right_channel_start, self.slice_end), (0, 255, 255), 2)
        cv2.line(frame, (self.right_channel_end, self.slice_start), 
                (self.right_channel_end, self.slice_end), (0, 255, 255), 2)
        
        # Always visualize the detected groups, even if not cutting
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        
        for i, group in enumerate(current_groups):
            color = colors[i % len(colors)]
            for point in group:
                # Add safety check to ensure point is valid
                try:
                    x, y = point
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        # Only draw points that are within the frame
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                except Exception as e:
                    print(f"Error drawing point {point}: {e}")
                    continue
        
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
                # Skip empty groups
                if not group:
                    continue
                    
                # Find closest previous group
                try:
                    group_center_x = sum(p[0] for p in group) / len(group)
                    
                    # Find the closest previous group by comparing centers
                    prev_centers = [sum(p[0] for p in g) / len(g) for g in self.previous_valid_groups if g]
                    if not prev_centers:
                        valid_groups.append(group)
                        continue
                        
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
                except Exception as e:
                    print(f"Error processing group: {e}")
                    continue
        else:
            # No previous groups to compare with, accept all
            valid_groups = [g for g in current_groups if g]
        
        # Update previous groups for next frame
        if valid_groups:
            self.previous_valid_groups = valid_groups.copy()
        
        # Process valid groups for galvo
        if valid_groups and self.is_cutting:
            # Clear previous points
            self.previous_points.clear()
            
            # Process each group in sequence
            for group_idx, group in enumerate(valid_groups):
                # Skip empty groups
                if not group:
                    continue
                    
                # Convert to galvo coordinates
                galvo_points = []
                for x, y in group:
                    # Skip invalid points
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        continue
                        
                    adjusted_x = x + self.galvo_offset_x
                    adjusted_y = y + self.galvo_offset_y + self.galvo_settings['offset_y']
                    
                    # Convert to hex coordinates
                    x_hex, y_hex = self.pixel_to_galvo_coordinates(adjusted_x, adjusted_y)
                    galvo_points.append((x_hex, y_hex))
                
                # Add points to processing queue with group metadata
                if galvo_points:
                    self.process_point_group(galvo_points, group_idx)
                
                # Draw the processed points
                color = (0, 0, 255)  # Red for valid cutting groups
                for point in group:
                    try:
                        x, y = point
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    except Exception as e:
                        print(f"Error highlighting point {point}: {e}")
                        continue
        
        # Update the frame queue
        self.calculate_loop_time()
        self.update_frame_queue(frame, frame_queue)
        
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