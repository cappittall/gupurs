import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import queue
import sys
import re

from laser_fabric_cutter import LaserFabricCutter

GALVO_SETTINGS_DEFAULT = {
                    'slice_size': 15,
                    'point_daviation': 3,
                    "white_area_index":3,
                    'offset_y': 0,
                    'travel_speed': 5000,
                    'frequency': 100,
                    'power': 50,
                    'cut_speed': 5000,
                    'laser_on_delay': 1,
                    'laser_off_delay': 1,
                    'polygon_delay': 50,
                    'offset_px': 25,  
                    'left_channel_start': 72,
                    'left_channel_end': 214,
                    'right_channel_start': 714,
                    'right_channel_end': 860
                }

class LaserCutterGUI:
    def __init__(self, master, video_source=None):
        self.settings_file = 'data/galvo_settings.json'
        self.load_galvo_settings(self.settings_file)
        self.master = master
        self.master.title("Laser Fabric Cutter")            
        try:
            with open('data/screen-size.txt', 'r') as f:
                screen_size = f.read().strip()
            # Validate screen size
            if not re.match(r'^\d+x\d+$', screen_size):
                raise ValueError("Invalid screen size format")
            width, height = map(int, screen_size.split('x'))
            if width <= 100 or height <= 100:
                raise ValueError("Window size too small")
        except (FileNotFoundError, ValueError):
            screen_size = "840x550"
    
  
        self.master.geometry(screen_size)  # Increased window size for better layout
        
        # Optionally, set minimum size to prevent the window from becoming too small
        #self.master.minsize(840, 550)

        self.cutter = None
        self.video_path = video_source
        self.is_cutting = False
        self.is_recording = False

        self.frame_queue = queue.Queue(maxsize=1)
        self.processing_thread = None

        self.video_writer = None
        self.recording_filename = None
                
        self.calibration_mode = False
        self.calibration_status = tk.StringVar(value="Offset Calibration: OFF \nStart - \"c\"")
        
        self.use_point_deviation = tk.BooleanVar(value=False)
        
        # Add key bindings
        self.master.bind('<c>', self.toggle_offset_calibration)
        self.master.bind('<w>', lambda event: self.adjust_galvo_offset(0, -1))
        self.master.bind('<s>', lambda event: self.adjust_galvo_offset(0, 1))
        self.master.bind('<a>', lambda event: self.adjust_galvo_offset(-1, 0))
        self.master.bind('<d>', lambda event: self.adjust_galvo_offset(1, 0))
        self.master.bind('<r>', self.reset_galvo_offset)        
        
        # Create a StringVar to hold the window size
        self.window_size_var = tk.StringVar()
        
        # Create the window size label
        self.window_size_label = ttk.Label(self.master, textvariable=self.window_size_var, anchor='se')
        self.window_size_label.pack(side='bottom', anchor='se', padx=5, pady=5)

        # Bind the configure event to update the window size
        self.master.bind('<Configure>', self.update_window_size)

        # Initial update of the window size
        self.update_window_size()
            
        self.create_styles()
        self.create_widgets()

        # Start the video stream immediately
        self.start_video_stream()
        
        
    def update_window_size(self, event=None):
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        self.window_size_var.set(f"Window size: {width}x{height}")
        if int(width) >100 and int(height) >100:
            with open('data/screen-size.txt', 'w') as f:
                f.write(f"{width}x{height}")
        
    def adjust_galvo_offset(self, dx, dy):
        if self.cutter and self.cutter.calibration_mode:
            self.cutter.adjust_galvo_offset(dx, dy)
            print(f"Galvo offset: X={self.cutter.galvo_offset_x}, Y={self.cutter.galvo_offset_y}")
    

    def reset_galvo_offset(self, event=None):
        if self.cutter and self.cutter.calibration_mode:
            self.cutter.galvo_offset_x = -45
            self.cutter.galvo_offset_y = 460
            print("Galvo offsets reset to initial values.")
            
    def create_styles(self):
            self.style = ttk.Style()
            
            # Create styles for Start and Stop buttons
            self.style.configure("Start.TButton", foreground="white", background="green")
            self.style.map("Start.TButton",
                        foreground=[('active', 'white')],
                        background=[('active', 'darkgreen')])
            
            self.style.configure("Stop.TButton", foreground="white", background="red")
            self.style.map("Stop.TButton",
                        foreground=[('active', 'white')],
                        background=[('active', 'darkred')])
    def create_widgets(self):
        # Main frame to hold everything
        main_frame = ttk.Frame(self.master)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Left frame (3/4 width) for image and main controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Right frame (1/4 width) for galvo status, sliders, and displays
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Configure the weights of the frames
        main_frame.columnconfigure(0, weight=3)  # Left frame
        main_frame.columnconfigure(1, weight=1)  # Right frame

        # Image display area
        self.image_canvas = tk.Canvas(left_frame)
        self.image_canvas.pack(expand=True, fill=tk.BOTH)
        self.image_canvas.bind("<Button-1>", self.canvas_click)
              
        # Calibration info label (moved to left_frame)
        calibration_info = ttk.LabelFrame(left_frame, text="Calibration info", width=400)
        calibration_info.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.calibration_info_label = tk.Text(calibration_info, height=5)
        self.calibration_info_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control buttons frame
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
         # Control buttons
        style = ttk.Style()
        style.configure("Start.TButton", foreground="white", background="green")
        self.start_stop_button = ttk.Button(control_frame, text="Start Cutting", command=self.toggle_cutting,  style="Start.TButton")
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
               
        self.aruco_calibrate_button = ttk.Button(control_frame, text="ArUco Calibrate", command=self.aruco_calibrate)
        self.aruco_calibrate_button.pack(side=tk.LEFT, padx=5)

        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # play/pause button to control frame
        self.play_pause_button = ttk.Button(control_frame, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)
          
        self.exit_button = ttk.Button(control_frame, text="Exit", command=self.exit_gui)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Right frame contents
        # Galvo connection status
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, pady=5)

        status_label = ttk.Label(status_frame, text="Galvo Status:")
        status_label.pack(side=tk.LEFT)
 
        self.connection_status = tk.Canvas(status_frame, width=20, height=20)
        self.connection_status.pack(side=tk.LEFT, padx=5)
        self.update_connection_status(False)

        # Sliders
        slider_frame = ttk.LabelFrame(right_frame, text="Galvo Settings")
        slider_frame.pack(fill=tk.X, pady=5)

        self.sliders = {}
        self.slider_values = {}
        for setting, value in self.settings.items():
            frame = ttk.Frame(slider_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            # channel or offset_px pass
            if setting in ['left_channel_start', 'left_channel_end', 'right_channel_start', 'right_channel_end', 'offset_px']:
                continue
            if setting == 'point_daviation':
                label = tk.Label(frame, text=setting.replace('_', ' ').title(), width=12)
                label.pack(side=tk.LEFT)
                # Add checkbox for point deviation
                self.point_deviation_checkbox = ttk.Checkbutton(
                    frame, 
                    variable=self.use_point_deviation,
                    command=self.toggle_point_deviation
                )
                self.point_deviation_checkbox.pack(side=tk.LEFT)
            else:
                label = ttk.Label(frame, text=setting.replace('_', ' ').title(), width=15)
                label.pack(side=tk.LEFT)
            
            value_label = ttk.Label(frame, text=str(value), width=8)
            value_label.pack(side=tk.RIGHT)
            
            slider = ttk.Scale(frame, from_=0, to=500000, orient=tk.HORIZONTAL, 
                            command=lambda v, s=setting, vl=value_label: self.update_setting(s, v, vl))
            
            slider.set(value)
            slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))
            
            self.sliders[setting] = slider
            self.slider_values[setting] = value_label

        # Display area
        display_frame = ttk.LabelFrame(right_frame, text="Status")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.display_area = tk.Text(display_frame, height=3, bg='black', fg='white')
        self.display_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Adjust slider ranges
        self.sliders['slice_size'].config(from_=15, to=100)
        self.sliders['white_area_index'].config(from_=1, to=5)
        self.sliders['point_daviation'].config(from_=1, to=100)
        self.sliders['offset_y'].config(from_=-200, to=200)
        # Adjust slider ranges
        self.sliders['travel_speed'].config(from_=1000, to=128000)
        self.sliders['cut_speed'].config(from_=1000, to=128000)
        self.sliders['frequency'].config(from_=20, to=200)
        self.sliders['power'].config(from_=0, to=100)
        self.sliders['laser_on_delay'].config(from_=0, to=100)    
        self.sliders['laser_off_delay'].config(from_=0, to=100)  
        self.sliders['polygon_delay'].config(from_=0, to=50)  

   
                    
        # Add calibration status label
        self.calibration_status_label = ttk.Label(right_frame, textvariable=self.calibration_status)
        self.calibration_status_label.pack(fill=tk.X, pady=5)
    
        # Add a new frame for channel selection in the right frame
        channel_frame = ttk.LabelFrame(right_frame, text="Channel Selection")
        channel_frame.pack(fill=tk.X, pady=5)
        
        # Left Channel
        left_channel_frame = ttk.Frame(channel_frame)
        left_channel_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(left_channel_frame, text="Left Channel", width=12).pack(side=tk.LEFT)
        
        left_start_frame = ttk.Frame(left_channel_frame)
        left_start_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(left_start_frame, text="Start", width=8).pack(side=tk.LEFT)
        self.left_start_value = ttk.Label(left_start_frame, text=str(self.settings['left_channel_start']), width=8)
        self.left_start_value.pack(side=tk.RIGHT)
        self.left_start_slider = ttk.Scale(left_start_frame, from_=0, to=1280, orient=tk.HORIZONTAL,
                            command=lambda v: self.update_channel_setting('left_channel_start', v, self.left_start_value))
        self.left_start_slider.set(self.settings['left_channel_start'])
        self.left_start_slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))
        
        left_end_frame = ttk.Frame(left_channel_frame)
        left_end_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(left_end_frame, text="End", width=8).pack(side=tk.LEFT)
        self.left_end_value = ttk.Label(left_end_frame, text=str(self.settings['left_channel_end']), width=8)
        self.left_end_value.pack(side=tk.RIGHT)
        self.left_end_slider = ttk.Scale(left_end_frame, from_=0, to=1280, orient=tk.HORIZONTAL,
                        command=lambda v: self.update_channel_setting('left_channel_end', v, self.left_end_value))
        self.left_end_slider.set(self.settings['left_channel_end'])
        self.left_end_slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))
        
        # Right Channel
        right_channel_frame = ttk.Frame(channel_frame)
        right_channel_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(right_channel_frame, text="Right Channel", width=12).pack(side=tk.LEFT)
        
        right_start_frame = ttk.Frame(right_channel_frame)
        right_start_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(right_start_frame, text="Start", width=8).pack(side=tk.LEFT)
        self.right_start_value = ttk.Label(right_start_frame, text=str(self.settings['right_channel_start']), width=8)
        self.right_start_value.pack(side=tk.RIGHT)
        self.right_start_slider = ttk.Scale(right_start_frame, from_=0, to=1280, orient=tk.HORIZONTAL,
                            command=lambda v: self.update_channel_setting('right_channel_start', v, self.right_start_value))
        self.right_start_slider.set(self.settings['right_channel_start'])
        self.right_start_slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))
        
        right_end_frame = ttk.Frame(right_channel_frame)
        right_end_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(right_end_frame, text="End", width=8).pack(side=tk.LEFT)
        self.right_end_value = ttk.Label(right_end_frame, text=str(self.settings['right_channel_end']), width=8)
        self.right_end_value.pack(side=tk.RIGHT)
        self.right_end_slider = ttk.Scale(right_end_frame, from_=0, to=1280, orient=tk.HORIZONTAL,
                            command=lambda v: self.update_channel_setting('right_channel_end', v, self.right_end_value))
        self.right_end_slider.set(self.settings['right_channel_end'])
        self.right_end_slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))
        
        # Add Offset PX slider
        offset_frame = ttk.Frame(channel_frame)
        offset_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(offset_frame, text="Offset PX", width=12).pack(side=tk.LEFT)
        self.offset_value = ttk.Label(offset_frame, text=str(self.settings['offset_px']), width=8)
        self.offset_value.pack(side=tk.RIGHT)
        self.offset_slider = ttk.Scale(offset_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                        command=lambda v: self.update_channel_setting('offset_px', v, self.offset_value))
        self.offset_slider.set(self.settings['offset_px'])
        self.offset_slider.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(0, 5))

    def update_channel_setting(self, setting_name, value, value_label):
        value = int(float(value))
        self.settings[setting_name] = value
        value_label.config(text=f"{value}")
        
        if self.cutter:
            # For offset_px, call the specific update method
            if setting_name == 'offset_px':
                self.cutter.update_offset_px(value)
            # For channel boundaries, update all boundaries at once to maintain consistency
            elif setting_name in ['left_channel_start', 'left_channel_end', 
                                'right_channel_start', 'right_channel_end']:
                self.cutter.update_channel_boundaries(
                    self.settings['left_channel_start'],
                    self.settings['left_channel_end'],
                    self.settings['right_channel_start'],
                    self.settings['right_channel_end']
                )
                
    def toggle_point_deviation(self):
        if self.cutter:
            self.cutter.use_point_deviation = self.use_point_deviation.get()
        print(f"Use Point Deviation: {self.use_point_deviation.get()}")
    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        if self.cutter:
            is_paused = self.cutter.toggle_pause()
            self.play_pause_button.config(text="Play" if is_paused else "Pause")
            
    def toggle_offset_calibration(self, event=None):
        self.calibration_mode = not self.calibration_mode
        status = "ON" if self.calibration_mode else "OFF"
        self.calibration_status.set(f"Offset Calibration: {status}  \nSave offset -\"c\"")
        if self.cutter:
            self.cutter.toggle_calibration_mode()
            print(f"Offset Calibration mode: {status}")
    
    def aruco_calibrate(self):
        if self.cutter:
            self.cutter.calibrate_cm_pixel_ratio()
            print("ArUco calibration completed")
                
    def load_galvo_settings(self, setting_file ):
        # Load settings from file if cutter is not initialized
        try:
            with open(setting_file, 'r') as f:
                self.settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Settings file not found or invalid. Using default settings.")
            # Define default settings
            self.settings = GALVO_SETTINGS_DEFAULT.copy()
            # Optionally, save the default settings back to the file
                
    def update_setting(self, setting_name, value, value_label):
        value = float(value)
        self.settings[setting_name] = value
        value_label.config(text=f"{value:.0f}")
        if self.cutter:
            self.cutter.update_galvo_settings(setting_name, float(value))
                    
    def canvas_click(self, event):
        if self.calibration_mode and self.cutter:
            x = event.x
            y = event.y
            self.cutter.adjust_galvo_offset(x - self.image_canvas.winfo_width() // 2,
                                            y - self.image_canvas.winfo_height() // 2)

    def update_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Ensure that the canvas size is greater than zero
        if canvas_width <= 0 or canvas_height <= 0:
            return  # Skip if the canvas has not been properly initialized

        img_width, img_height = image.size
        aspect_ratio = img_width / img_height

        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)

        # Ensure the new dimensions are valid
        if new_width > 0 and new_height > 0:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
            self.image_canvas.image = photo

        
    def start_video_stream(self):
        if not self.cutter:
            self.cutter = LaserFabricCutter(self.video_path, self.settings, self.settings_file)

        self.cutter.start_processing(self.frame_queue)
        self.master.after(100, self.update_gui)

    def update_connection_status(self, is_connected):
        color = "green" if is_connected else "red"
        self.connection_status.create_oval(2, 2, 18, 18, fill=color, outline="")

    def toggle_cutting(self):
        if not self.cutter:
            print("Video path:", self.video_path)
            if not self.video_path:
                return

        if not self.is_cutting:
            self.is_cutting = True
            self.cutter.start_cutting()
        else:
            self.is_cutting = False
            self.cutter.stop_cutting()

        self.update_button_appearance()

    def update_button_appearance(self):
        if self.is_cutting:
            self.start_stop_button.configure(text="Stop Cutting", style="Stop.TButton")
        else:
            self.start_stop_button.configure(text="Start Cutting", style="Start.TButton")

    def process_video(self):
        self.cutter.process_video(self.frame_queue)
    
    def update_gui(self):
        try:
            frame = self.frame_queue.get_nowait()
            self.update_image(frame)
            
            # Update display area
            self.display_area.delete('1.0', tk.END)
            self.display_area.insert(tk.END, f" Width: {self.cutter.width:.2f} cm\n")
            self.display_area.insert(tk.END, f" Speed: {self.cutter.fabric_speed:.2f} cm/min\n")
            self.display_area.insert(tk.END, f" Image Loop Time: {self.cutter.point_loop_time:.4f} s\n")
            self.display_area.insert(tk.END, f" Galvo Loop Time: {self.cutter.galvo_loop_time:.4f} s\n")
            self.display_area.insert(tk.END, f" Number of Points: {self.cutter.point_queue.qsize()} ")
            
            # Automatically check the point deviation checkbox when cutting starts
            if self.cutter.fabric_speed > 0:
                pass
                #self.use_point_deviation.set(True)
            else:
                pass
                #self.use_point_deviation.set(False)

            self.point_deviation_checkbox.update()  # Refresh the checkbox state

            if self.cutter:
                #self.display_calibration_info()
                self.calibration_info_label.delete('1.0', tk.END)
                self.calibration_info_label.insert(tk.END, f" Pixel cm ratio: {self.cutter.pixel_cm_ratio}\n\n")
                self.calibration_info_label.insert(tk.END, f" Galvo Hex: ({self.cutter.current_x_hex}, {self.cutter.current_y_hex})\n")
                self.calibration_info_label.insert(tk.END, f" Galvo Pixel: ({self.cutter.adjusted_x}, {self.cutter.adjusted_y})\n")
                self.calibration_info_label.insert(tk.END, f" Current_x, fixed_y: ({self.cutter.point_x}, {self.cutter.point_y})\n")
                self.calibration_info_label.insert(tk.END, f" Galvo Offset: ({self.cutter.galvo_offset_x}, {self.cutter.galvo_offset_y})")

            # Update connection status
            self.update_connection_status(self.cutter.galvo_connection)

        except queue.Empty:
            pass

        self.master.after(30, self.update_gui)   
            
    def calibrate(self):
        if self.cutter:
            self.cutter.calibrate(self.cutter.get_current_frame())

    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            file_path = filedialog.asksaveasfilename(
                defaultextension=".avi",
                filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
            )
            if file_path:
                self.is_recording = True
                self.recording_filename = file_path
                self.record_button.config(text="Stop Recording")
                # VideoWriter will be initialized when the next frame is available
            else:
                # User cancelled, do not start recording
                pass
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.recording_filename = None
                
    def reset(self):
        if self.cutter:
            self.cutter = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording_filename = None
        self.is_cutting = False
        self.is_recording = False
        self.start_stop_button.config(text="Start Cutting")
        self.record_button.config(text="Start Recording")
        self.update_connection_status(False)
        self.display_area.delete('1.0', tk.END)
        self.display_area.insert(tk.END, "Width: 0.00 cm\nSpeed: 0.00 cm/min\nLoop Time: 0.0000 s")

    def exit_gui(self):
        try:
            if self.is_cutting:
                self.is_cutting = False
                self.cutter.stop_cutting()
            
            if self.cutter:
                # Set a timeout for cleanup operations
                cleanup_thread = threading.Thread(target=self._cleanup_with_timeout)
                cleanup_thread.start()
                cleanup_thread.join(timeout=3.0)  # Wait max 3 seconds
                
            if self.video_writer:
                self.video_writer.release()
                
            self.master.quit()
            self.master.destroy()
            os._exit(0)  # Force exit if normal shutdown fails
            
        except Exception as e:
            print(f"Forcing exit due to: {e}")
            os._exit(1)

    def _cleanup_with_timeout(self):
        try:
            self.cutter.stop_processing()
            self.cutter.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")

 
if __name__ == "__main__":
    video_source = 0  # Default to camera
    
  # Check command line arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--video" and i + 1 < len(sys.argv):
            video_source = sys.argv[i + 1]
            print(f"Using video file: {video_source}")
        elif arg == "--image" and i + 1 < len(sys.argv):
            video_source = {"type": "image", "path": sys.argv[i + 1]}
            print(f"Using static image: {video_source['path']}")
        elif arg.endswith((".mp4", ".avi")):  # Direct video file argument
            video_source = arg
            print(f"Using video file: {video_source}")
        elif arg.endswith((".jpg", ".jpeg", ".png")):  # Direct image file argument
            video_source = {"type": "image", "path": arg}
            print(f"Using static image: {video_source['path']}")
    
    try:
        root = tk.Tk()
        app = LaserCutterGUI(root, video_source=video_source)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        if app:
            app.exit_gui()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if app:
            app.exit_gui()