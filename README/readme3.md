# Balor Galvo Laser Control Module

## Overview

This module provides a Python interface for controlling BJJCZ (Golden Orange, Beijing JCZ) LMCV4-FIBER-M and compatible galvo laser cutter boards. It consists of two main classes: `Sender` and `CommandList`, which work together to create and execute laser cutting jobs.

## Features

- USB communication with the laser cutter board
- Creation and execution of laser cutting jobs
- Support for various laser operations (cutting, traveling, setting parameters)
- Simulation capabilities for previewing jobs
- Extensible command system

## Installation

To use this module, you need to have the following dependencies installed:

- Python 3.6+
- pyusb

You can install the required dependencies using pip:


pip install pyusb


## Usage

### Sender Class

The `Sender` class handles communication with the laser cutter board and executes jobs.

```python
from balor_laser import Sender

# Initialize the Sender
sender = Sender()

# Open connection to the laser cutter
sender.open()

# Create a job
job = sender.job()

# Add commands to the job
# ...

# Execute the job
sender.execute(job)

# Close the connection
sender.close()

CommandList Class
The CommandList class represents a laser cutting job and provides methods for adding various operations.

# Create a job
job = sender.job()

# Add operations to the job
job.travel(x, y)
job.cut(x, y)
job.set_laser_power(power)
job.set_cut_speed(speed)
# ...

# Execute the job
sender.execute(job)

Main Classes
Sender
The Sender class is responsible for:

Establishing USB connection with the laser cutter board
Initializing the machine
Executing jobs
Handling low-level communication with the board
Key Methods:
open(): Opens the connection to the laser cutter
close(): Closes the connection
execute(command_list, loop_count=1): Executes a job
abort(): Aborts the current job
Various raw_* methods for low-level board operations
CommandList
The CommandList class represents a laser cutting job and provides methods for:

Adding various laser operations (cut, travel, set parameters)
Generating command packets for execution
Simulating the job for preview purposes
Key Methods:
travel(x, y): Move the laser head without cutting
cut(x, y): Move the laser head while cutting
set_laser_power(power): Set the laser power
set_cut_speed(speed): Set the cutting speed
serialize(): Generate the binary representation of the job
packet_generator(): Generate command packets for execution
Operation Classes
The module includes various Operation subclasses representing different laser cutter commands, such as:

OpTravel: Move without cutting
OpCut: Move while cutting
OpSetCutSpeed: Set cutting speed
OpMarkPowerRatio: Set laser power
OpLaserControl: Turn laser on/off
These classes are used internally by the CommandList to represent individual operations in a job.

Simulation
The module includes a Simulation class that can be used to preview jobs without actually running them on the laser cutter. This is useful for testing and visualization purposes.

License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Disclaimer
This software is provided as-is, without any warranty. Use at your own risk. Always follow proper safety procedures when operating laser cutting equipment.

Contributing
Contributions to this project are welcome. Please submit pull requests or open issues on the project's GitHub repository.

Authors
Gnostic Instruments, Inc.
Acknowledgments
This module is based on reverse-engineering efforts of the BJJCZ (Golden Orange, Beijing JCZ) LMCV4-FIBER-M laser cutter board protocol.


This README provides an overview of the Balor Galvo Laser Control Module, including its main classes, usage examples, installation instructions, and other relevant information. You may want to adjust some details based on the specific implementation and any additional features or requirements of your project.