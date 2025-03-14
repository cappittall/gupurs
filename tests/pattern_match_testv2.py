import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PatternDetector:
    def __init__(self, video_path, output_dir=None):
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Parameters
        self.slice_start = 300
        self.slice_height = 40
        self.detection_threshold = 0.6
        
        # Create windows for visualization
        #cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 600, 700)
        
        # Parameters with default values
        self.params = {
            'threshold1': 100,
            'threshold2': 180,
            'blur_size': 3,
            'clahe_clip': 2,
            'clahe_grid': 8,
            'morph_kernel': 3,
            'dilate_iter': 1,
            'erode_iter': 1,
            'min_area': 100,
            'min_height_ratio': 50,  # as percentage
            'gap_threshold': 30,
            'method': 0
        }
        
        # Create controls
        self.setup_controls()
        
    def setup_controls(self):
        # Basic parameters
        cv2.createTrackbar('Slice Start', 'Controls', self.slice_start, 600, self.update_slice_start)
        cv2.createTrackbar('Slice Height', 'Controls', self.slice_height, 100, self.update_slice_height)
        cv2.createTrackbar('Threshold 1', 'Controls', self.params['threshold1'], 255, 
                          lambda x: self.update_param('threshold1', x))
        cv2.createTrackbar('Threshold 2', 'Controls', self.params['threshold2'], 255, 
                          lambda x: self.update_param('threshold2', x))
        
        # Preprocessing parameters
        cv2.createTrackbar('Blur Size', 'Controls', self.params['blur_size'], 15, 
                          lambda x: self.update_param('blur_size', max(1, x if x % 2 == 1 else x + 1)))
        cv2.createTrackbar('CLAHE Clip', 'Controls', self.params['clahe_clip']*10, 50, 
                          lambda x: self.update_param('clahe_clip', x/10))
        cv2.createTrackbar('CLAHE Grid', 'Controls', self.params['clahe_grid'], 16, 
                          lambda x: self.update_param('clahe_grid', max(1, x)))
        
        # Morphology parameters
        cv2.createTrackbar('Morph Kernel', 'Controls', self.params['morph_kernel'], 15, 
                          lambda x: self.update_param('morph_kernel', max(1, x if x % 2 == 1 else x + 1)))
        cv2.createTrackbar('Dilate Iter', 'Controls', self.params['dilate_iter'], 10, 
                          lambda x: self.update_param('dilate_iter', x))
        cv2.createTrackbar('Erode Iter', 'Controls', self.params['erode_iter'], 10, 
                          lambda x: self.update_param('erode_iter', x))
        
        # Filtering parameters
        cv2.createTrackbar('Min Area', 'Controls', self.params['min_area'], 1000, 
                          lambda x: self.update_param('min_area', x))
        cv2.createTrackbar('Min Height %', 'Controls', self.params['min_height_ratio'], 100, 
                          lambda x: self.update_param('min_height_ratio', x))
        cv2.createTrackbar('Gap Threshold', 'Controls', self.params['gap_threshold'], 100, 
                          lambda x: self.update_param('gap_threshold', x))
        
        # Detection method
        cv2.createTrackbar('Method', 'Controls', self.params['method'], 4, 
                        lambda x: self.update_param('method', x))
    
    def update_param(self, param_name, value):
        self.params[param_name] = value
        
    def update_slice_start(self, value):
        self.slice_start = value
        
    def update_slice_height(self, value):
        self.slice_height = value
    
    def process_video(self):
        # Open the video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Add a button for saving parameters
        cv2.createTrackbar('Save Settings', 'Controls', 0, 1, self.save_parameters)
        
        quit_requested = False
        frame_idx = 0
        paused = False
        last_frame = None
        
        while not quit_requested:
            if not paused:
                # Reset video to beginning if needed
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                last_frame = frame.copy()
                frame_idx += 1
            else:
                # Use the last frame when paused
                frame = last_frame.copy() if last_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process the frame
            start_time = time.time()
            result_img = self.process_frame(frame.copy())
            processing_time = time.time() - start_time
            
            # Add frame info
            cv2.putText(result_img, f"Frame: {frame_idx}, Time: {processing_time:.3f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show parameter values on the result image
            y_pos = 60
            for param, value in self.params.items():
                if param != 'method':  # Skip method as it's shown differently
                    cv2.putText(result_img, f"{param}: {value}", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
            
            # Show results
            #cv2.imshow('Original', frame)
            cv2.imshow('Processed', result_img)
            
            # Save output if requested
            if self.output_dir and frame_idx % 30 == 0 and not paused:
                output_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(output_path, result_img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                quit_requested = True
            elif key == ord('s'):
                # Save current frame
                if self.output_dir:
                    output_path = os.path.join(self.output_dir, f"snapshot_{frame_idx:04d}.jpg")
                    cv2.imwrite(output_path, result_img)
                    print(f"Saved snapshot to {output_path}")
            elif key == ord('p'):
                # Toggle pause
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('n') and paused:
                # Step forward one frame when paused
                ret, frame = cap.read()
                if ret:
                    last_frame = frame.copy()
                    frame_idx += 1
            
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def save_parameters(self, x):
        if x == 1:  # Only save when trackbar is moved to position 1
            # Create settings dict
            settings = {
                'slice_start': self.slice_start,
                'slice_height': self.slice_height,
                **self.params
            }
            
            # Save to file
            if self.output_dir:
                params_path = os.path.join(self.output_dir, "detection_params.txt")
                with open(params_path, 'w') as f:
                    for key, value in settings.items():
                        f.write(f"{key}: {value}\n")
                print(f"Parameters saved to {params_path}")
            
            # Reset trackbar
            cv2.setTrackbarPos('Save Settings', 'Controls', 0)
    
    def process_frame(self, frame):
        """Process a frame to detect patterns using multiple methods with adjustable parameters"""
        # Create a composite image to show all processing stages
        h, w = frame.shape[:2]
        result = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Draw boundary lines for the slice
        slice_start = self.slice_start
        slice_end = slice_start + self.slice_height
        cv2.line(frame, (0, slice_start), (w, slice_start), (255, 0, 0), 1)
        cv2.line(frame, (0, slice_end), (w, slice_end), (255, 0, 0), 1)
        
        # Get the slice
        slice_img = frame[slice_start:slice_end, :]
        if slice_img.size == 0:  # Check if slice is valid
            print(f"Invalid slice: start={slice_start}, height={self.slice_height}")
            return frame
        
        # Stage 1: Original frame with slice highlighted
        stage1_img = frame.copy()
        result[0:h, 0:w] = stage1_img
        
        # Stage 2: Preprocessing pipeline
        # Convert to grayscale
        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur (use odd kernel size)
        blur_size = self.params['blur_size']
        if blur_size > 0:
            slice_blurred = cv2.GaussianBlur(slice_gray, (blur_size, blur_size), 0)
        else:
            slice_blurred = slice_gray
            
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip'], 
            tileGridSize=(self.params['clahe_grid'], self.params['clahe_grid'])
        )
        slice_enhanced = clahe.apply(slice_blurred)
        
        # Threshold to get binary image
        _, slice_binary = cv2.threshold(
            slice_enhanced, 
            self.params['threshold1'], 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Apply morphological operations
        kernel_size = self.params['morph_kernel']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if self.params['erode_iter'] > 0:
            slice_binary = cv2.erode(slice_binary, kernel, iterations=self.params['erode_iter'])
        
        if self.params['dilate_iter'] > 0:
            slice_binary = cv2.dilate(slice_binary, kernel, iterations=self.params['dilate_iter'])
        
        # Convert to 3 channels for display
        slice_binary_display = cv2.cvtColor(slice_binary, cv2.COLOR_GRAY2BGR)
        stage2_img = np.zeros_like(frame)
        stage2_img[slice_start:slice_end, :] = slice_binary_display
        
        # Copy to result
        result[0:h, w:w*2] = stage2_img
        
        # Stage 3: Edge detection
        edges = cv2.Canny(
            slice_enhanced, 
            self.params['threshold1'], 
            self.params['threshold2']
        )
        
        # Apply morphological operations to edges for better continuity
        if self.params['dilate_iter'] > 0:
            edges = cv2.dilate(edges, kernel, iterations=self.params['dilate_iter'])
            
        if self.params['erode_iter'] > 0:
            edges = cv2.erode(edges, kernel, iterations=self.params['erode_iter'])
        
        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        stage3_img = np.zeros_like(frame)
        stage3_img[slice_start:slice_end, :] = edges_display
        
        # Copy to result
        result[h:h*2, 0:w] = stage3_img
        
        # Stage 4: Detection method based on method_idx
        stage4_img = frame.copy()
        method_idx = self.params['method']

        if method_idx == 0:
            # Method 1: Contour-based detection
            detected_points = self.detect_using_contours(slice_binary, slice_start)
        elif method_idx == 1:
            # Method 2: Hough Line-based detection
            detected_points = self.detect_using_hough_lines(edges, slice_start)
        elif method_idx == 2:
            # Method 3: Morphological operations
            detected_points = self.detect_using_morphology(slice_binary, slice_enhanced, slice_start)
        elif method_idx == 3:
            # Method 4: Gradient-based edge detection with filtering
            detected_points = self.detect_using_gradient(slice_binary, slice_enhanced, slice_start)
        else:
            # Method 5: Canny edge with S-contour detection
            detected_points = self.detect_using_canny_contours(slice_binary, slice_enhanced, slice_start)

            
        # Group detected points
        grouped_points = self.group_points(detected_points, self.params['gap_threshold'])
        
        # Draw grouped points with different colors
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        
        for i, group in enumerate(grouped_points):
            color = colors[i % len(colors)]
            
            # Draw points
            for x, y in group:
                cv2.circle(stage4_img, (int(x), int(y)), 3, color, -1)
            
            # Draw lines connecting points in each group if there are enough points
            if len(group) > 1:
                # Sort points by y-coordinate
                sorted_points = sorted(group, key=lambda p: p[1])

                
                # Optionally apply spline smoothing for cleaner visualization
                if len(sorted_points) >= 4:
                    pts = self.fit_spline_safely(sorted_points)
                    if pts is not None:
                        cv2.polylines(stage4_img, [pts], False, color, 2)
                    else:
                        # Fallback to simple line segments if spline fails
                        for j in range(len(sorted_points) - 1):
                            pt1 = (int(sorted_points[j][0]), int(sorted_points[j][1]))
                            pt2 = (int(sorted_points[j+1][0]), int(sorted_points[j+1][1]))
                            cv2.line(stage4_img, pt1, pt2, color, 2)
                else:
                    # If too few points for spline, just connect with lines
                    for j in range(len(sorted_points) - 1):
                        pt1 = (int(sorted_points[j][0]), int(sorted_points[j][1]))
                        pt2 = (int(sorted_points[j+1][0]), int(sorted_points[j+1][1]))
                        cv2.line(stage4_img, pt1, pt2, color, 2)
            
            # Label the group
            if group:
                avg_x = sum(p[0] for p in group) / len(group)
                cv2.putText(stage4_img, f"G{i+1}", (int(avg_x), slice_start - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
        # Copy to result
        result[h:h*2, w:w*2] = stage4_img
        
        # Add method name
        method_names = ["Contour", "Hough Lines", "Morphology", "Gradient", "S-Contour"]
        cv2.putText(result, f"Method: {method_names[min(method_idx, len(method_names)-1)]}", 
                    (w+10, h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detected group count
        cv2.putText(result, f"Groups: {len(grouped_points)}", 
                    (w+10, h+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        return result
    
    def detect_using_contours(self, binary_img, slice_start):
        """Detect pattern points using contour analysis with enhanced filtering"""
        points = []
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine minimum height requirement based on percentage
        min_height = binary_img.shape[0] * (self.params['min_height_ratio'] / 100.0)
        
        # Filter and process contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.params['min_area']:  # Minimum area threshold
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only consider contours that span most of the slice height
            if h < min_height:
                continue
                
            # Extract centerline points
            for row in range(binary_img.shape[0]):
                # Get the horizontal slice of the contour at this row
                mask = np.zeros_like(binary_img)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                row_pixels = mask[row, :]
                
                if np.any(row_pixels):
                    # Find the leftmost and rightmost white pixels
                    leftmost = np.where(row_pixels > 0)[0][0]
                    rightmost = np.where(row_pixels > 0)[0][-1]
                    
                    # Calculate center point
                    center_x = (leftmost + rightmost) // 2
                    center_y = row + slice_start
                    
                    points.append((center_x, center_y))
        
        return points
    
    def detect_using_hough_lines(self, edges, slice_start):
        """Detect pattern points using Hough Line Transform with filtering"""
        points = []
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                               threshold=20, minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            # Group lines by proximity and angle
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Filter for near-vertical lines (S/Z shape might have slight angles)
                if angle > 60:  # Close to vertical (90 degrees)
                    vertical_lines.append((x1, y1 + slice_start, x2, y2 + slice_start))
                    
                    # Add points along the line
                    if y2 > y1:
                        for y in range(y1, y2 + 1, 2):
                            # Interpolate x based on line equation
                            t = (y - y1) / max(1, y2 - y1)
                            x = int(x1 + t * (x2 - x1))
                            points.append((x, y + slice_start))
                    else:
                        for y in range(y2, y1 + 1, 2):
                            # Interpolate x based on line equation
                            t = (y - y2) / max(1, y1 - y2)
                            x = int(x2 + t * (x1 - x2))
                            points.append((x, y + slice_start))
        
        return points
    
    def detect_using_morphology(self, binary_img, enhanced_img, slice_start):
        """Detect pattern points using morphological operations"""
        points = []
        
        # Get kernel for morphological operations
        kernel_size = self.params['morph_kernel']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply morphological operations to isolate pattern edges
        opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find vertical edges by applying gradient in x direction
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        gradient_x = cv2.filter2D(closing, -1, kernel_x)
        gradient_x = cv2.convertScaleAbs(gradient_x)
        
        # Threshold the gradient
        _, edges = cv2.threshold(gradient_x, self.params['threshold1'], 255, cv2.THRESH_BINARY)
        
        # Dilate edges to connect nearby points
        dilated = cv2.dilate(edges, kernel, iterations=self.params['dilate_iter'])
        
        # Find contours on the dilated edges
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine minimum height requirement
        min_height = binary_img.shape[0] * (self.params['min_height_ratio'] / 100.0)
        
        # Process each contour to find centerline points
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.params['min_area']:
                continue
                
            # Only consider contours that span most of the slice height
            if h < min_height:
                continue
                
            # Extract centerline points
            for row in range(binary_img.shape[0]):
                # Get the horizontal slice of the contour at this row
                mask = np.zeros_like(binary_img)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                row_pixels = mask[row, :]
                
                if np.any(row_pixels):
                    # Find the leftmost and rightmost white pixels
                    leftmost = np.where(row_pixels > 0)[0][0]
                    rightmost = np.where(row_pixels > 0)[0][-1]
                    
                    # Calculate center point
                    center_x = (leftmost + rightmost) // 2
                    center_y = row + slice_start
                    
                    points.append((center_x, center_y))
        
        return points
    
    def detect_using_gradient(self, binary_img, enhanced_img, slice_start):
        """Detect pattern edges using gradient-based methods with additional filtering"""
        points = []
        
        # Apply Sobel operator in x direction to detect vertical edges
        sobel_x = cv2.Sobel(enhanced_img, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        
        # Threshold the gradient image
        _, edge_binary = cv2.threshold(abs_sobel_x, self.params['threshold1'], 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to enhance edges
        kernel = np.ones((self.params['morph_kernel'], self.params['morph_kernel']), np.uint8)
        dilated = cv2.dilate(edge_binary, kernel, iterations=self.params['dilate_iter'])
        eroded = cv2.erode(dilated, kernel, iterations=self.params['erode_iter'])
        
        # Find contours on the processed edges
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine minimum height requirement
        min_height = binary_img.shape[0] * (self.params['min_height_ratio'] / 100.0)
        
        # Process valid contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.params['min_area']:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only consider contours that span most of the slice height
            if h < min_height:
                continue
                
            # Calculate contour's perimeter to area ratio
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Filter for elongated shapes (indicative of vertical channels)
                if circularity < 0.5:  # More elongated shapes have lower circularity
                    # Extract points along the contour's centerline
                    for row in range(binary_img.shape[0]):
                        # Create a mask for this contour
                        mask = np.zeros_like(binary_img)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        
                        # Get pixels in this row
                        row_pixels = mask[row, :]
                        
                        if np.any(row_pixels):
                            # Find the leftmost and rightmost white pixels
                            leftmost = np.where(row_pixels > 0)[0][0]
                            rightmost = np.where(row_pixels > 0)[0][-1]
                            
                            # Calculate center point
                            center_x = (leftmost + rightmost) // 2
                            center_y = row + slice_start
                            
                            points.append((center_x, center_y))
        
        return points
    
    def detect_using_canny_contours(self, binary_img, enhanced_img, slice_start):
        """Detect S-shaped patterns using a combination of Canny edges and contour filtering"""
        points = []
        
        # Apply Canny edge detection with adjustable thresholds
        edges = cv2.Canny(enhanced_img, 
                        self.params['threshold1'], 
                        self.params['threshold2'])
        
        # Apply morphological operations to connect disconnected edges
        kernel = np.ones((self.params['morph_kernel'], self.params['morph_kernel']), np.uint8)
        if self.params['dilate_iter'] > 0:
            edges = cv2.dilate(edges, kernel, iterations=self.params['dilate_iter'])
        if self.params['erode_iter'] > 0:
            edges = cv2.erode(edges, kernel, iterations=self.params['erode_iter'])
        
        # Find contours on the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum height requirement based on percentage of slice height
        min_height = binary_img.shape[0] * (self.params['min_height_ratio'] / 100.0)
        
        # Filter and analyze S-shaped contours
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < self.params['min_area']:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by height to ensure we have vertical structures
            if h < min_height:
                continue
            
            # Check for S-shape characteristics
            # 1. Calculate contour perimeter and shape complexity
            perimeter = cv2.arcLength(contour, True)
            shape_complexity = perimeter**2 / (4 * np.pi * area)
            
            # S-shapes typically have higher complexity ratio
            # (adjust threshold based on your specific patterns)
            if shape_complexity < 2.0:  # More complex shapes have higher values
                continue
                
            # 2. Check for inflection points typical in S-curves
            # Fit a minimum area rectangle to check orientation
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Analyze contour curvature to detect S-shape
            # Extract centerline points along the contour
            for row in range(binary_img.shape[0]):
                # Create a horizontal scan line
                mask = np.zeros_like(binary_img)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                row_pixels = mask[row, :]
                
                if np.any(row_pixels):
                    # Find the leftmost and rightmost white pixels
                    leftmost = np.where(row_pixels > 0)[0][0]
                    rightmost = np.where(row_pixels > 0)[0][-1]
                    
                    # Calculate center point
                    center_x = (leftmost + rightmost) // 2
                    center_y = row + slice_start
                    
                    points.append((center_x, center_y))
        
        return points

    
    def fit_spline_safely(self, points):
        """Safely fit a spline to a set of points, with extensive error handling"""
        if len(points) < 4:
            return None
        
        try:
            # Extract x and y coordinates
            x_coords = np.array([float(p[0]) for p in points])
            y_coords = np.array([float(p[1]) for p in points])
            
            # Check for NaN or infinite values
            if np.any(np.isnan(x_coords)) or np.any(np.isnan(y_coords)) or \
            np.any(np.isinf(x_coords)) or np.any(np.isinf(y_coords)):
                return None
            
            # Check for sufficient number of unique points
            if len(set(x_coords)) < 3 or len(set(y_coords)) < 3:
                return None
            
            # Use a much higher smoothing factor to avoid warnings
            # The smoothing factor should scale with the number of points and their variance
            s_factor = max(len(points), np.var(x_coords) + np.var(y_coords))
            
            # Create parametric spline with adaptive smoothing
            tck, u = splprep([x_coords, y_coords], 
                            s=s_factor, 
                            k=min(3, len(points)-1))
            
            # Sample more points for a smoother curve
            u_new = np.linspace(0, 1, num=len(points)*2)
            
            # Evaluate the spline
            x_new, y_new = splev(u_new, tck)
            
            # Filter out any invalid results
            valid_mask = np.logical_and(np.isfinite(x_new), np.isfinite(y_new))
            x_new = x_new[valid_mask]
            y_new = y_new[valid_mask]
            
            # Check if we have enough valid points after filtering
            if len(x_new) < 2:
                return None
            
            # Create points array and cast safely
            # Round values before casting to int to avoid warnings
            points_array = np.column_stack((np.round(x_new), np.round(y_new)))
            pts = points_array.reshape((-1, 1, 2)).astype(np.int32)
            
            return pts
        
        except Exception as e:
            print(f"Spline fitting error: {e}")
            return None

    def group_points(self, points, x_threshold=30):
        """Group points by their x-coordinate proximity with improved clustering"""
        if not points:
            return []
        
        # Sort points by x-coordinate
        sorted_points = sorted(points, key=lambda p: p[0])
        
        # Find point clusters using the x-coordinate
        x_values = np.array([p[0] for p in sorted_points])
        
        # If very few points, return as a single group
        if len(x_values) < 5:
            return [sorted_points]
        
        # More advanced clustering technique
        # Calculate pairwise distances between adjacent x values
        diffs = np.diff(x_values)
        
        # If all points are close together, return as a single group
        if np.max(diffs) < x_threshold:
            return [sorted_points]
        
        # Use histogram-based method to identify clusters
        hist, bin_edges = np.histogram(x_values, bins=min(20, len(x_values)//2))
        
        # Find significant gaps - bins with zero or very few points
        gap_indices = []
        
        # Look for significant gaps in x-coordinate distribution

        # Look for significant gaps in x-coordinate distribution
        for i in range(len(diffs)):
            # If the gap is larger than threshold, this is a boundary between groups
            if diffs[i] > x_threshold:
                gap_indices.append(i)
        
        # Create groups based on gap indices
        groups = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            groups.append(sorted_points[start_idx:gap_idx+1])
            start_idx = gap_idx + 1
            
        # Add the final group
        if start_idx < len(sorted_points):
            groups.append(sorted_points[start_idx:])
        
        # Try to ensure we have close to 4 groups (for 4 pattern edges)
        # If too many groups, merge the smallest adjacent ones
        while len(groups) > 4:
            # Find the pair of adjacent groups with smallest combined size
            smallest_pair_idx = -1
            smallest_pair_size = float('inf')
            
            for i in range(len(groups) - 1):
                combined_size = len(groups[i]) + len(groups[i+1])
                if combined_size < smallest_pair_size:
                    smallest_pair_size = combined_size
                    smallest_pair_idx = i
            
            # Merge the smallest pair
            if smallest_pair_idx >= 0:
                groups[smallest_pair_idx] = groups[smallest_pair_idx] + groups[smallest_pair_idx+1]
                groups.pop(smallest_pair_idx+1)
            else:
                break
        
        # If too few groups, try to split the largest ones
        attempts = 0
        while len(groups) < 4 and attempts < 3:
            # Find the largest group
            largest_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
            largest_group = groups[largest_idx]
            
            # Only try to split if it's big enough
            if len(largest_group) >= 10:
                # Sort by x-coordinate
                sorted_group = sorted(largest_group, key=lambda p: p[0])
                x_coords = [p[0] for p in sorted_group]
                
                # Find the largest gap within the group
                diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
                if diffs:
                    max_gap_idx = np.argmax(diffs)
                    
                    # If the gap is significant, split the group
                    if diffs[max_gap_idx] > x_threshold/2:
                        group1 = sorted_group[:max_gap_idx+1]
                        group2 = sorted_group[max_gap_idx+1:]
                        
                        # Replace the original group with the two new groups
                        groups.pop(largest_idx)
                        groups.append(group1)
                        groups.append(group2)
                        
                        # Sort groups by average x-coordinate
                        groups.sort(key=lambda g: sum(p[0] for p in g)/len(g) if g else 0)
            
            attempts += 1
        
        return groups

def main():
    parser = argparse.ArgumentParser(description='Enhanced Pattern Detection Test Tool')
    parser.add_argument('--video', type=str, default='data/5.webm', help='Path to input video file')
    parser.add_argument('--output', type=str, default='data/pattern_test_output', help='Path to output directory')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use instead of video file')
    args = parser.parse_args()
    
    # Use camera if specified, otherwise use video file
    video_source = args.camera if args.camera is not None else args.video
    
    detector = PatternDetector(video_source, args.output)
    detector.process_video()

if __name__ == "__main__":
    main()

