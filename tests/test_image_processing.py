import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_image_processing(image_path):
    # Read and process image
    frame = cv2.imread(image_path)
    slice_start = 320
    slice_height = 100
    slice_end = slice_start + slice_height
    slice_img = frame[slice_start:slice_end, :]
    # Create figure for visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(331)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    # Step 1: Slice and convert to gray
    slice_img = frame[slice_start:slice_end, :]
    gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
    plt.subplot(332)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Slice')
    
    # Step 2: Gaussian blur
    blur = cv2.GaussianBlur(gray, (27, 27), 5)
    plt.subplot(333)
    plt.imshow(blur, cmap='gray')
    plt.title('Gaussian Blur')
    
    # Step 3: Sobel edges
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.subplot(334)
    plt.imshow(sobel_combined_normalized, cmap='gray')
    plt.title('Sobel Edges')
    
    # Step 4: Threshold
    _, thresholded = cv2.threshold(sobel_combined_normalized, 50, 255, cv2.THRESH_BINARY)
    plt.subplot(335)
    plt.imshow(thresholded, cmap='gray')
    plt.title('Thresholded')
    
    # Step 5: Morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, vertical_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vertical_kernel)
    plt.subplot(336)
    plt.imshow(opened, cmap='gray')
    plt.title('Morphological Operations')
    
    # Step 6: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    plt.subplot(337)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title('Connected Components')
    
    # Step 7: Size filtering
    sizes = stats[1:, -1]
    min_size = 50
    cleaned = np.zeros(opened.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            cleaned[labels == i] = 255
            
    plt.subplot(338)
    plt.imshow(cleaned, cmap='gray')
    plt.title('Size Filtered')
    
    # Step 8: Find S-shape borders
    def find_s_shape_borders(image, target_area_index=3):
        height, width = image.shape[:2]
        points = []
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
                            points.append((x, y))
                            break
                else:
                    in_white_area = False
        return points

    # Find and visualize borders
    result_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    points = find_s_shape_borders(cleaned)
    if points:
        for point in points:
            cv2.circle(result_img, point, 3, (0, 255, 0), -1)
            
    plt.subplot(339)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected S-shape Borders')
    
    plt.tight_layout()
    plt.show()
    
    return points

if __name__ == "__main__":
    image_path = "data/guipure_pattern.jpeg"
    detected_points = test_image_processing(image_path)
    print(f"Detected {len(detected_points)} points along S-shape borders")
