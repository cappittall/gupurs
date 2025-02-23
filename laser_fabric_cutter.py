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
        self.s_shape_border = None
        
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

        self.threshold_percent = 15  # Default threshold percentage (adjustable)
        self.initial_point_count = 10  # Number of points to collect initially
        self.initial_point_established = False  # Flag to indicate if initial point is set
        
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

    def find_s_shape_bordersX(self, image):
        height, width = image.shape[:2]
        points = []
        for y in range(height):
    
            x_range = range(0, width)
            white_area_count = 0
            in_white_area = False
            for x in x_range:
                pixel_value = image[y, x]
                if pixel_value > 0:
                    if not in_white_area:
                        white_area_count += 1
                        in_white_area = True
                        if white_area_count == 3:
                            # Found the 3rd white area
                            adjusted_y = y + self.slice_start
                            points.append((x, adjusted_y))
                            break  # Move to next row after finding the point
                else:
                    in_white_area = False  # We are in a black area

        if not points:
            print("3rd white area not found in any rows.")
            return None
        else:
            return points
    
    def find_s_shape_borders(self, image):
        height, width = image.shape[:2]
        points = []
        target_area_index = round(float(self.galvo_settings.get('white_area_index', 3)))

        for y in range(height):
            white_area_count = 0
            in_white_area = False

            for x in range(width):
                pixel_value = image[y, x]
                if pixel_value > 0:
                    if not in_white_area:
                        white_area_count += 1
                        in_white_area = True
                        if white_area_count == target_area_index:
                            # Found the target white area
                            adjusted_y = y + self.slice_start
                            points.append((x, adjusted_y))
                            break  # Move to the next row after finding the point
                else:
                    in_white_area = False  # Reset when leaving the white area

        if not points:
            print(f"{target_area_index}th white area not found in any rows.")
            return None
        return points

                
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
                    
                    
                # Disable autofocus for Logitech C920
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 disables autofocus
                self.cap.set(cv2.CAP_PROP_FOCUS, 50)     # Set a fixed focus value (0-255, adjust as needed)
                # Camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                
                while self.cap.isOpened() and self.is_running:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        if isinstance(self.video_path, str):  # Video file
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                            continue
                        else:  # Camera error
                            print("Frame capture failed")
                            break

                    self.process_frame(frame.copy(), frame_queue)
                    time.sleep(0.01)
                    
        except Exception as e:
            print("Error in image processing:", e)
            traceback.print_exc()
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

    # TODO
    def process_frame(self, frame, frame_queue): 
        min_contour_area = 500      # Minimum contour area to consider
        offset_px = 25               # Laser offset in pixels
        blur_size = 5               # Size of Gaussian blur kernel

        slice_img = frame[self.slice_start:self.slice_end, :]
        # Image processing steps
        try:
           # Convert to grayscale and blur
            gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                # Process contours
            s_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter by area and height
                if area > min_contour_area and h > self.slice_height*0.8:
                    s_contours.append(cnt)
                    
            # Create point arrays
            laser_points = []
            for cnt in s_contours:
                # Get centerline points
                points = []
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                for y in range(self.slice_height):
                    row = mask[y,:]
                    if np.any(row):
                        left = np.argmax(row)
                        right = len(row) - np.argmax(row[::-1]) - 1
                        cx = (left + right) // 2
                        points.append((cx, y))
                
                # Apply alternating offset
                offset_points = []
                for i, (x, y) in enumerate(points):
                    if i % 2 == 0:
                        offset_points.append((x + offset_px, y))
                    else:
                        offset_points.append((x - offset_px, y))
                
                laser_points.append(offset_points)
                
            colored_slice_bgr = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
            for points in laser_points:
                for x, y in points:
                    cv2.circle(colored_slice_bgr, (x, y), 5, (0,0,255), -1)
                    
                    
                    
        
        except Exception as e:
            logging.error(f"Error -1  {e}")
            return
            
            
        self.calculate_loop_time()
        
        h = colored_slice_bgr.shape[0]
        slice_height = min(h, self.slice_height)
        frame[self.slice_start:self.slice_start+slice_height, :] = colored_slice_bgr[:slice_height]
                        
        # In process_frame method
        if laser_points:
            new_points = set()  # Set to store new unique points
            for contour_points in laser_points:  # Iterate through each contour's points
                for point in contour_points:  # Iterate through points in each contour
                    self.point_x, self.point_y = point  # Now we correctly unpack each (x,y) tuple
                    self.point_y = self.point_y + self.slice_start
                    self.adjusted_x = self.point_x + self.galvo_offset_x
                    self.adjusted_y = self.point_y + self.galvo_offset_y + self.galvo_settings['offset_y']

                    # Convert to hexadecimal coordinates
                    x_hex, y_hex = self.pixel_to_galvo_coordinates(self.adjusted_x, self.adjusted_y)
                    point = (x_hex, y_hex)

                    # Check if point is new and valid
                    if point not in self.previous_points and self.is_valid_point(self.point_x, self.point_y):
                        new_points.add(point)
                        #cv2.circle(frame, (int(self.point_x), int(self.point_y)), 5, (0, 255, 0), -1)

            # Add new unique points to the queue
            for point in new_points:
                self.send_point_to_galvo(*point)
            
            # Update previous_points with the new points        
            self.previous_points = new_points

        else:
            logging.debug("3rd white area not found. Skipping this frame.")
            # Reset width, current_x, and previous_points when no border is found
            self.width = 0
            self.current_x = None
            self.adjusted_y = None
            self.previous_points.clear()


        # Always update the frame, even if no border is found
        self.update_frame_queue(frame, frame_queue)

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
    # galvo_cutting_loop implementation
    def galvo_cutting_loop(self):
        if not self.sender:
            logging.error("Sender is not initialized")
            return

        first_move = True  # Track if it's the first move

        # Helper function to process points
        def process_points(points, params, cutting):
            if not points:
                return  # Skip if no points

            if first_move:
                self.sender.set_xy(points[0][0], points[0][1])  # Move to the first point
                print(f"First move to: {points[0][0]}, {points[0][1]}")

            # Tick function to control laser cutting
            def tick(cmds, loop_index):
                cmds.clear()
                cmds.set_mark_settings(**params)  # Apply laser settings

                for point in points:
                    if cutting:  # Mark the points only if cutting
                        cmds.mark(point[0], point[1])
                    else:  # Move to the points without marking
                        self.sender.set_xy(point[0], point[1])
                       

            # Create and execute the laser job
            job = self.sender.job(tick=tick)
            job.execute(1)

        # Include relevant keys for laser settings
        include_keys = [
            'travel_speed', 'frequency', 'power', 'cut_speed',
            'laser_on_delay', 'laser_off_delay', 'polygon_delay'
        ]

        while self.is_running and not self.stop_event.is_set():
            # Gather current laser settings
            params = {k: v for k, v in self.galvo_settings.items() if k in include_keys}
            start_time = time.monotonic()  # Track loop start time

            try:
                # Get all queued points
                points = self.get_all_points()
                if points:
                    # Determine if the system is in cutting mode
                    cutting = self.is_cutting

                    # Process points in a separate thread
                    threading.Thread(target=process_points, args=(points, params, cutting)).start()
                    first_move = False  # Mark the first move as done

            except Exception as e:
                logging.error(f"Error in galvo_loop: {e}")

            # Adjust loop timing to maintain smooth operation
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