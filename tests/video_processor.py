from time import sleep
import cv2
import numpy as np

def process_movie(movie_path):
    while True:  # Infinite loop to restart video when finished
        cap = cv2.VideoCapture(movie_path)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Crop horizontal slice from y=200 to 240
            slice_y_start = 200
            slice_y_end = 240
            slice_frame = frame[slice_y_start:slice_y_end, :]
            
            # Process only the slice
            img = cv2.cvtColor(slice_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # 1. Preprocessing
            kernel_size = (9, 9)
            blurred = cv2.GaussianBlur(img, kernel_size, 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # 2. Segmentation (Otsu's Thresholding)
            ret, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 3. Morphological Operations
            morph_kernel = np.ones((3,3), np.uint8)
            closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, morph_kernel)

            # 4. Contour Detection
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 5. Filter Contours (based on area, shape, etc.)
            s_channels = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    # Add more filtering criteria (e.g., aspect ratio, rectangularity) here
                    s_channels.append(cnt)

            # Draw the contours on the grayscale slice for visualization
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_color, s_channels, -1, (0, 0, 255), 2)  # Red contours

            # Create visualization with original frame and add guide lines showing the slice
            display_frame = frame.copy()
            cv2.line(display_frame, (0, slice_y_start), (frame.shape[1], slice_y_start), (0, 255, 0), 1)
            cv2.line(display_frame, (0, slice_y_end), (frame.shape[1], slice_y_end), (0, 255, 0), 1)
            
            # Prepare all processing stages for display - convert grayscale images to color
            blurred_colored = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            enhanced_colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            thresholded_colored = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            closed_colored = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            
            # Add labels to each image - keep original width
            stages = [
                (slice_frame, "Original Slice"),
                (blurred_colored, "Blurred"),
                (enhanced_colored, "Enhanced"),
                (thresholded_colored, "Thresholded"),
                (closed_colored, "Closed"),
                (img_color, "Contours (Red)")
            ]
            
            labeled_images = []
            for img, label in stages:
                labeled = img.copy()
                cv2.putText(labeled, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                labeled_images.append(labeled)
            
            # Stack all images vertically - no resizing
            processing_stages = np.vstack(labeled_images)
            
            # Display the original frame and processing stages
            cv2.imshow("Original Frame with Slice Highlighted", display_frame)
            cv2.imshow("Processing Stages", processing_stages)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit the function completely
            sleep(0.1)

        # Release the current capture before starting again
        cap.release()


# Example Usage
if __name__ == "__main__":
    process_movie("data/5.webm")






""" import cv2
import numpy as np
import os

def process_frame(frame):
    # Extract the region of interest (slice)
    region = frame[200:220, :]
    
    # Create a copy of original region for visualization
    original = region.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to strengthen lines
    dilation_kernel = np.ones((10, 1), np.uint8)
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=1)
    
    # Use morphological closing to connect broken lines
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, closing_kernel)
    
    # Create a horizontal kernel to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,20))
    horizontal_lines = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Find contours
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a visualization with contours
    contour_vis = np.zeros_like(region)
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
    
    # Get coordinates of contours for laser cutting
    cutting_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            # Store as (x, y+200) to map back to original frame coordinates
            cutting_points.append((x, y+200))
    
    # Create the final visualization with all steps
    # Convert grayscale images to BGR for stacking
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Also convert blurred to BGR - this was missing
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    dilated_edges_bgr = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    closed_edges_bgr = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
    horizontal_lines_bgr = cv2.cvtColor(horizontal_lines, cv2.COLOR_GRAY2BGR)
    
    # Create labels for each processing step
    labels = ["Original", "Grayscale", "Blurred", "Edges", 
              "Dilated Edges", "Closed Edges", "Horizontal Lines", "Contours"]
    
    # Combine all steps vertically
    result_images = [original, gray_bgr, blurred_bgr, edges_bgr, 
                    dilated_edges_bgr, closed_edges_bgr, 
                    horizontal_lines_bgr, contour_vis]
    
    # Add labels to each image
    for i, img in enumerate(result_images):
        cv2.putText(img, labels[i], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Stack images vertically with padding
    rows = []
    for img in result_images:
        # Add padding for better visibility
        padded = cv2.copyMakeBorder(img, 5, 5, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        rows.append(padded)
    
    visualization = np.vstack(rows)
    
    return visualization, cutting_points

def main():
    # Open video file
    # Change the path to be relative to the project root
    video_path = 'data/5.webm'
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print(f"Current directory: {os.getcwd()}")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Control variables
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            frame_idx += 1
        
        # Process the current frame
        visualization, cutting_points = process_frame(frame)
        
        # Display processing steps
        cv2.imshow('Video Processing Steps', visualization)
        
        # Display cutting points on original frame
        result_frame = frame.copy()
        for x, y in cutting_points:
            cv2.circle(result_frame, (x, y), 3, (0, 0, 255), -1)
        
        # Draw horizontal line to show the slice region
        cv2.line(result_frame, (0, 200), (frame_width, 200), (0, 255, 0), 1)
        cv2.line(result_frame, (0, 220), (frame_width, 220), (0, 255, 0), 1)
        
        # Show frame number
        cv2.putText(result_frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Original with Cutting Points', result_frame)
        
        # Control playback
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == 32:  # 'p' or spacebar
            paused = not paused
        elif key == ord('n') and paused:  # 'n' - next frame when paused
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 """