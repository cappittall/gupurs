# Balor Galvo Laser Control Module - Instruction Guide

## Overview

This document provides instructions on how to use the `Sender` and `CommandList` classes from the Balor Galvo Laser Control Module for fabric cutting applications. These classes facilitate communication with BJJCZ (Beijing JCZ) LMCV4-FIBER-M laser controllers and compatible boards, allowing you to define and execute laser cutting jobs based on detected points.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [1. Initialize and Connect to the Laser Machine](#1-initialize-and-connect-to-the-laser-machine)
  - [2. Create a Cutting Job](#2-create-a-cutting-job)
  - [3. Set Laser Parameters](#3-set-laser-parameters)
  - [4. Define the Cutting Path](#4-define-the-cutting-path)
  - [5. Execute the Cutting Job](#5-execute-the-cutting-job)
  - [6. Close the Connection](#6-close-the-connection)
- [Advanced Usage](#advanced-usage)
  - [Simulation (Optional)](#simulation-optional)
  - [Handling Laser Output](#handling-laser-output)
  - [Error Handling](#error-handling)
- [Tips and Best Practices](#tips-and-best-practices)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Prerequisites

- **Python 3.x** installed on your system.
- **USB Communication Permissions**: Ensure your user has permissions to access USB devices (may require administrative privileges or adjustments to udev rules on Linux).
- **Laser Safety Knowledge**: Familiarity with laser operation safety protocols.

## Installation

1. **Clone or Download the Repository** containing the `Sender` and `CommandList` classes.

2. **Install Required Python Packages**:

   ```bash
   pip install pyusb numpy
   ```

---

## Getting Started

### 1. Initialize and Connect to the Laser Machine

- **Objective**: Establish a connection with the laser machine using the `Sender` class.
- **Steps**:
  - Import the `Sender` class.
  - Create an instance of `Sender`.
  - Use the `open()` method to connect to the laser machine.

- **Example**:

  ```python
  from sender import Sender

  # Create a Sender instance
  sender = Sender(debug=True)  # Enable debug mode if needed

  # Open connection to the laser machine (machine_index may vary)
  sender.open(machine_index=0)
  ```

### 2. Create a Cutting Job

- **Objective**: Create a job that will contain the sequence of commands for the laser.
- **Steps**:
  - Use the `job()` method from the `Sender` instance to create a `CommandList` object.

- **Example**:

  ```python
  # Create a CommandList (job)
  job = sender.job()
  ```

### 3. Set Laser Parameters

- **Objective**: Configure the laser settings required for the cutting operation.
- **Parameters to Set**:
  - **Travel Speed**: Speed at which the laser moves when not cutting.
  - **Cut Speed**: Speed at which the laser moves when cutting.
  - **Laser Power**: Power of the laser beam (in percentage).
  - **Frequency**: Laser pulse frequency (in kHz).
  - **Delays**: Various delays like laser on/off delay, polygon delay.

- **Example**:

  ```python
  # Set laser parameters
  job.set_mark_settings(
      travel_speed=2000,      # Travel speed in mm/s
      frequency=20,           # Laser frequency in kHz
      power=50,               # Laser power in %
      cut_speed=1000,         # Cutting speed in mm/s
      laser_on_delay=100,     # Laser on delay in us
      laser_off_delay=100,    # Laser off delay in us
      polygon_delay=50        # Polygon delay in us
  )
  ```

### 4. Define the Cutting Path

- **Objective**: Add commands to the job to define the cutting path based on detected points.
- **Steps**:
  - Loop through your list of detected points.
  - Use `mark(x, y)` to add cutting commands to the job.

- **Example**:

  ```python
  # Assume detected_points is a list of (x, y) tuples
  detected_points = [(x1, y1), (x2, y2), (x3, y3), ...]

  # Move to the starting point without firing the laser
  start_point = detected_points[0]
  job.goto(*start_point)

  # Turn on the laser control
  job.laser_control(True)

  # Loop through points and add cutting commands
  for point in detected_points:
      x, y = point
      job.mark(x, y)

  # Turn off the laser control after cutting
  job.laser_control(False)
  ```

### 5. Execute the Cutting Job

- **Objective**: Send the job to the laser machine for execution.
- **Steps**:
  - Use the `execute()` method from the `Sender` instance.
  - Specify the number of times the job should be executed with `loop_count`.

- **Example**:

  ```python
  # Execute the job once
  sender.execute(job, loop_count=1)
  ```

### 6. Close the Connection

- **Objective**: Safely close the connection to the laser machine after the job is completed.
- **Steps**:
  - Use the `close()` method from the `Sender` instance.

- **Example**:

  ```python
  # Close the connection
  sender.close()
  ```

---

## Advanced Usage

### Simulation (Optional)

- **Objective**: Simulate the cutting job to visualize the path before actual execution.
- **Requirements**:
  - `Pillow` library for image creation and manipulation.

- **Installation**:

  ```bash
  pip install Pillow
  ```

- **Example**:

  ```python
  from PIL import Image, ImageDraw
  from command_list import Simulation

  # Create an image for simulation
  image_size = 2048  # Adjust as needed
  image = Image.new('RGB', (image_size, image_size), 'white')
  draw = ImageDraw.Draw(image)

  # Create a Simulation instance
  sim = Simulation(job, sender, draw, resolution=image_size)

  # Plot the job onto the image
  job.plot(draw)

  # Save or display the simulation image
  image.save('simulation.png')
  # image.show()  # Uncomment to display the image
  ```

### Handling Laser Output

- **Turning the Laser On/Off**:

  ```python
  # Turn laser on
  job.laser_control(True)

  # Turn laser off
  job.laser_control(False)
  ```

- **Moving Without Firing the Laser**:

  ```python
  # Move to position without firing the laser
  job.goto(x, y)
  ```

- **Controlling Output Ports** (e.g., for accessories like lights):

  ```python
  # Turn on a port (e.g., port 8 for light)
  job.port_on(bit=8)

  # Turn off a port
  job.port_off(bit=8)
  ```

### Error Handling

- **Exceptions**: Be prepared to handle exceptions that may occur during communication or execution.

- **Example**:

  ```python
  try:
      sender.open()
      # ... execute job ...
  except BalorCommunicationException as e:
      print(f"Communication error: {e}")
  except BalorMachineException as e:
      print(f"Machine error: {e}")
  finally:
      sender.close()
  ```

---

## Tips and Best Practices

- **Safety First**: Always follow laser safety protocols. Wear appropriate protective equipment and ensure the laser area is secure.
- **Testing**: Before cutting actual fabric, test the job on a sample material to verify the settings and path.
- **Permissions**: Ensure you have the necessary permissions to access USB devices. On Linux systems, you may need to adjust udev rules or run the script with `sudo`.
- **Debugging**: Use the `debug=True` parameter when initializing the `Sender` to enable debug messages.
- **Calibration**: If using calibration data (`cal` parameter), ensure it accurately reflects your laser setup for precise cutting.

---

## License

This software is licensed under the GNU General Public License v3.0. See the `LICENSE` file for details.

---

## Acknowledgements

- **Gnostic Instruments, Inc.** for the original codebase.
- **BJJCZ (Beijing JCZ)** for the LMCV4-FIBER-M laser controller.
- **Contributors** who have helped improve and maintain the code.

---

*This instruction guide is intended to assist users in setting up and operating the Balor Galvo Laser Control Module for fabric cutting applications. For any issues or further assistance, please refer to the project's documentation or contact the support team.*