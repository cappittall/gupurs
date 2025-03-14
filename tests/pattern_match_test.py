import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import argparse

class PatternDetector:
    def __init__(self, video_path, output_dir=None):
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Parameters
        self.slice_start = 300  # Adjustable
        self.slice_height = 40
        self.detection_threshold = 0.6
        
        # Create windows for visualization
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
        
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
        
        # Create trackbars for parameter adjustment
        cv2.createTrackbar('Slice Start', 'Processed', self.slice_start, height, self.update_slice_start)
        cv2.createTrackbar('Slice Height', 'Processed', self.slice_height, 100, self.update_slice_height)
        cv2.createTrackbar('Threshold 1', 'Processed', 100, 255, lambda x: None)
        cv2.createTrackbar('Threshold 2', 'Processed', 180, 255, lambda x: None)
        cv2.createTrackbar('Method', 'Processed', 0, 3, lambda x: None)
        
        quit_requested = False
        frame_idx = 0
        
        while not quit_requested:
            # Reset video to beginning if needed
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_idx += 1
            
            # Get current parameter values
            threshold1 = cv2.getTrackbarPos('Threshold 1', 'Processed')
            threshold2 = cv2.getTrackbarPos('Threshold 2', 'Processed')
            method_idx = cv2.getTrackbarPos('Method', 'Processed')
            
            # Process the frame
            start_time = time.time()
            result_img = self.process_frame(frame.copy(), threshold1, threshold2, method_idx)
            processing_time = time.time() - start_time
            
            # Add frame info
            cv2.putText(result_img, f"Frame: {frame_idx}, Time: {processing_time:.3f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show results
            cv2.imshow('Original', frame)
            cv2.imshow('Processed', result_img)
            
            # Save output if requested
            if self.output_dir and frame_idx % 10 == 0:
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
            
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def update_slice_start(self, value):
        self.slice_start = value
        
    def update_slice_height(self, value):
        self.slice_height = value
    
    def process_frame(self, frame, threshold1, threshold2, method_idx):
        """Process a frame to detect patterns using multiple methods"""
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
        
        # Stage 1: Original frame with slice highlighted
        stage1_img = frame.copy()
        # Copy to result
        result[0:h, 0:w] = stage1_img
        
        # Stage 2: Basic preprocessing
        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        slice_enhanced = clahe.apply(slice_gray)
        
        # Threshold to get binary image
        _, slice_binary = cv2.threshold(slice_enhanced, threshold1, 255, cv2.THRESH_BINARY_INV)
        
        # Convert to 3 channels for display
        slice_binary_display = cv2.cvtColor(slice_binary, cv2.COLOR_GRAY2BGR)
        stage2_img = np.zeros_like(frame)
        stage2_img[slice_start:slice_end, :] = slice_binary_display
        
        # Copy to result
        result[0:h, w:w*2] = stage2_img
        
        # Stage 3: Edge detection
        edges = cv2.Canny(slice_enhanced, threshold1, threshold2)
        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        stage3_img = np.zeros_like(frame)
        stage3_img[slice_start:slice_end, :] = edges_display
        
        # Copy to result
        result[h:h*2, 0:w] = stage3_img
        
        # Stage 4: Detection method based on method_idx
        stage4_img = frame.copy()
        
        if method_idx == 0:
            # Method 1: Contour-based detection
            detected_points = self.detect_using_contours(slice_binary, slice_start)
        elif method_idx == 1:
            # Method 2: Hough Line-based detection
            detected_points = self.detect_using_hough_lines(edges, slice_start)
        elif method_idx == 2:
            # Method 3: Morphological operations
            detected_points = self.detect_using_morphology(slice_binary, slice_start)
        else:
            # Method 4: Watershed algorithm
            detected_points = self.detect_using_watershed(slice_img, slice_binary, slice_start)
            
        # Group detected points
        grouped_points = self.group_points(detected_points)
        
        # Draw grouped points with different colors
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        
        for i, group in enumerate(grouped_points):
            color = colors[i % len(colors)]
            # Draw points
            for x, y in group:
                cv2.circle(stage4_img, (int(x), int(y)), 3, color, -1)
            
            # Draw lines connecting points in each group
            if len(group) > 1:
                # Sort points by y-coordinate
                sorted_points = sorted(group, key=lambda p: p[1])
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
        method_names = ["Contour", "Hough Lines", "Morphology", "Watershed"]
        cv2.putText(result, f"Method: {method_names[method_idx]}", 
                    (w+10, h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        return result
    
    def detect_using_contours(self, binary_img, slice_start):
        """Detect pattern points using contour analysis"""
        points = []
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only consider contours that span most of the slice height
            if h < binary_img.shape[0] * 0.5:
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
        """Detect pattern points using Hough Line Transform"""
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
                
                # Filter for near-vertical lines
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
    

    def detect_using_morphology(self, binary_img, slice_start):
        """Detect pattern points using morphological operations"""
        points = []
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find vertical edges by applying gradient in x direction
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        gradient_x = cv2.filter2D(closing, -1, kernel_x)
        gradient_x = cv2.convertScaleAbs(gradient_x)
        
        # Threshold the gradient
        _, edges = cv2.threshold(gradient_x, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate edges to connect nearby points
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours on the dilated edges
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour to find centerline points
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only consider contours that span most of the slice height
            if h < binary_img.shape[0] * 0.5:
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
    
    def detect_using_watershed(self, slice_img, binary_img, slice_start):
        """Detect pattern points using watershed segmentation"""
        points = []
        
        # Convert to BGR for watershed
        markers = np.zeros(binary_img.shape, dtype=np.int32)
        
        # Define background and foreground markers
        # Background
        markers[0, :] = 1
        markers[-1, :] = 1
        markers[:, 0] = 1
        markers[:, -1] = 1
        
        # Foreground - use distance transform to find sure foreground
        dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        sure_fg_labels = cv2.connectedComponents(sure_fg)[1]
        markers[sure_fg > 0] = sure_fg_labels[sure_fg > 0] + 1
        
        # Apply watershed
        cv2.watershed(cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB), markers)
        
        # Extract boundaries between regions
        boundary_mask = np.zeros_like(binary_img)
        boundary_mask[markers == -1] = 255
        
        # Find vertical boundaries
        for col in range(boundary_mask.shape[1]):
            col_pixels = boundary_mask[:, col]
            if np.any(col_pixels):
                # Find all boundary points in this column
                boundary_rows = np.where(col_pixels > 0)[0]
                for row in boundary_rows:
                    points.append((col, row + slice_start))
        
        return points
    
    def group_points(self, points, x_threshold=30):
        """Group points by their x-coordinate proximity"""
        if not points:
            return []
        
        # Sort points by x-coordinate
        sorted_points = sorted(points, key=lambda p: p[0])
        
        # Find point clusters using the x-coordinate
        x_values = np.array([p[0] for p in sorted_points])
        
        # Identify significant gaps in x-values
        gaps = [x_values[i+1] - x_values[i] for i in range(len(x_values)-1)]
        if not gaps:
            return [sorted_points]
        
        # Calculate median and standard deviation of gaps
        median_gap = np.median(gaps)
        std_gap = np.std(gaps)
        
        # Set threshold for significant gaps (median + 2*std is a common approach)
        significant_gap = max(x_threshold, median_gap + 2*std_gap)
        
        # Find indices where significant gaps occur
        gap_indices = [i for i, gap in enumerate(gaps) if gap > significant_gap]
        
        # Create groups based on gap indices
        groups = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            groups.append(sorted_points[start_idx:gap_idx+1])
            start_idx = gap_idx + 1
            
        # Add the final group
        if start_idx < len(sorted_points):
            groups.append(sorted_points[start_idx:])
        
        return groups

def main():
    parser = argparse.ArgumentParser(description='Pattern Detection Test Tool')
    parser.add_argument('--video', type=str, default='data/5.webm', help='Path to input video file')
    parser.add_argument('--output', type=str, default='data/pattern_test_output', help='Path to output directory')
    args = parser.parse_args()
    
    detector = PatternDetector(args.video, args.output)
    detector.process_video()

if __name__ == "__main__":
    main()
  
