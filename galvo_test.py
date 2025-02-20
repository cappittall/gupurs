import cv2
import numpy as np
import time
import os
import sys

# Update the system path to include the balor module
sys.path.append('../balor')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'balor')))
sys.path.append('/home/yordam/balor')
galvo_connected = False

try:
    # Import the necessary galvo control modules
    from balor.sender import Sender  # type: ignore 
    from balor.command_list import OpSetTravelSpeed, OpSetQSwitchPeriod, OpTravel, OpCut, OpMarkPowerRatio, OpLaserControl  # type: ignore
    print('Balor object loaded...')
except Exception as e:
    print(f'Error loading Balor object: {e}')
    print(f'Current sys.path: {sys.path}')
    print(f'Current working directory: {os.getcwd()}')
    sys.exit(1)

# Initialize the galvo
try:
    sender = Sender()
    # Load the correction table if necessary
    cor_table_data = open("tools/jetsonCalibrationdeneme1.cor", 'rb').read()
    if hasattr(sender, 'set_cor_table'):
        sender.set_cor_table(cor_table_data)
    galvo_connected = sender.open()
    if galvo_connected:
        print("Galvo connected successfully.")
    else:
        print("Failed to connect to galvo.")
       
except Exception as e:
    print(f"Error initializing galvo: {e}")
    

# Create an 'inspect' folder if it doesn't exist
os.makedirs('inspect', exist_ok=True)

# Define image parameters
image_size = 512  # Define the size of the image
step_size = 10  # The amount by which the square size decreases each iteration

# Open live camera feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    cap = cv2.VideoCapture(1)

# Function to map pixel coordinates to galvo hex values
def pixel_to_galvo_hex(xy_pixel):
    x_pixel, y_pixel = xy_pixel
    # Map the pixel coordinates to galvo coordinate range (0x0000 to 0xFFFF)
    x_galvo = int((x_pixel / image_size) * 0xFFFF)
    y_galvo = int((y_pixel / image_size) * 0xFFFF)
    x_galvo_hex = f"{x_galvo:04X}"
    y_galvo_hex = f"{y_galvo:04X}"
    return x_galvo, y_galvo, x_galvo_hex, y_galvo_hex

# Main loop to draw squares and move the galvo
square_size = image_size
while square_size > 0:
    # Capture a frame from the live camera feed
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Resize the frame to match the image size
    image = cv2.resize(frame, (image_size, image_size))
    center = (image_size // 2, image_size // 2)  # Center of the image

    # Calculate the top-left and bottom-right coordinates of the square
    top_left = (center[0] - square_size // 2, center[1] - square_size // 2)
    bottom_right = (center[0] + square_size // 2, center[1] + square_size // 2)
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # Draw the square on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Get galvo hex values for the four corners of the square
    corners = [top_left, top_right, bottom_right, bottom_left]
    for i, corner in enumerate(corners):
        x, y, x_galvo_hex, y_galvo_hex = pixel_to_galvo_hex(corner)
        corner_text = f"Corner {i+1}: {corner}, Galvo: ({x_galvo_hex}, {y_galvo_hex})"
        cv2.putText(image, corner_text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Move the galvo to draw the square (if connected)
    if galvo_connected:
        cmds = sender.job()
        cmds.clear()
        # Set travel speed and other parameters if needed
        cmds.append(OpSetTravelSpeed(1000))  # Adjust speed as necessary
        # Move to each corner of the square
        cmds.append(OpTravel(*pixel_to_galvo_hex(top_left)[:2], 0, 0))
        cmds.append(OpCut(*pixel_to_galvo_hex(top_right)[:2], 0, 0))
        cmds.append(OpCut(*pixel_to_galvo_hex(bottom_right)[:2], 0, 0))
        cmds.append(OpCut(*pixel_to_galvo_hex(bottom_left)[:2], 0, 0))
        cmds.append(OpCut(*pixel_to_galvo_hex(top_left)[:2], 0, 0))
        # Execute the commands
        sender.execute(cmds)
        print(f"Galvo moved to draw square with size {square_size}")
    else:
        print("Galvo not connected, skipping movement.")

    # Display the image
    cv2.imshow('Galvo Test', image)

    # Wait for a key press
    key = cv2.waitKey(1000)  # Wait for 1000 ms (1 second)
    if key == ord('s'):
        # Save the image if 's' is pressed
        filename = f"inspect/galvo_test_{square_size}.png"
        cv2.imwrite(filename, image)
        print(f"Image saved to {filename}")
        
    elif key == 27:  # ESC key to exit
        break

    # Decrease the square size for the next iteration
    square_size -= step_size

# Release camera and close the galvo connection
cap.release()
if galvo_connected:
    sender.close()
cv2.destroyAllWindows()
print("Galvo test completed.")
