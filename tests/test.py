import cv2
import numpy as np

def test_image_processing(image_path):
    # Adjustable parameters
    crop_y = 320                # Starting Y position for cropping
    crop_height = 100           # Height of the cropped area
    threshold_value = 120       # Initial threshold value (0-255)
    min_contour_area = 500      # Minimum contour area to consider
    offset_px = 20               # Laser offset in pixels
    blur_size = 5               # Size of Gaussian blur kernel
    invert_threshold = True     # Invert threshold for dark shapes
    
    # Read and crop image
    img = cv2.imread(image_path)
    crop = img[crop_y:crop_y+crop_height, :]
    orig_display = crop.copy()
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Thresholding
    if invert_threshold:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours
    s_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter by area and height
        if area > min_contour_area and h > crop_height*0.8:
            s_contours.append(cnt)
    
    # Create point arrays
    laser_points = []
    for cnt in s_contours:
        # Get centerline points
        points = []
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        for y in range(crop_height):
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
    
    # Visualization
    cv2.imshow('1. Original Crop', crop)
    cv2.imshow('2. Grayscale', gray)
    cv2.imshow('3. Thresholded', thresh)
    
    # Draw contours
    contour_display = crop.copy()
    cv2.drawContours(contour_display, s_contours, -1, (0,255,0), 2)
    cv2.imshow('4. Detected Contours', contour_display)
    
    # Draw laser path
    path_display = crop.copy()
    for points in laser_points:
        for x, y in points:
            cv2.circle(path_display, (x, y), 1, (0,0,255), -1)
    cv2.imshow('5. Laser Path Preview', path_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_image_processing("data/guipure_pattern.jpeg")
