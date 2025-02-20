import cv2
import numpy as np

def create_moving_fabric_video(image_path, output_path, duration_seconds=10, fps=30):
    # Read the source image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image from {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames and pixels to shift per frame
    total_frames = duration_seconds * fps
    pixels_per_frame = 2  # Adjust this for different speeds
    
    # Create each frame
    for i in range(total_frames):
        # Calculate shift amount
        shift = (i * pixels_per_frame) % height
        
        # Create new frame by rolling the image
        new_frame = np.roll(img, -shift, axis=0)
        
        # Write frame to video
        out.write(new_frame)
    
    out.release()

# Usage
image_path = "data/guipure_pattern.jpeg"
output_path = "data/moving_fabric.avi"
create_moving_fabric_video(image_path, output_path)
