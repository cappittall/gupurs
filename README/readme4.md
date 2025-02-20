# Balor Galvo Laser Control Module

The Balor Galvo Laser Control Module is a Python library designed to interface with BJJCZ (Golden Orange, Beijing JCZ) LMCV4-FIBER-M and compatible laser control boards. It provides a simplified interface for controlling laser operations, including movements, power settings, and job execution.

This module is intended for use in laser cutting and engraving applications, allowing for precise control of laser parameters and movements through a high-level API.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Initializing the Sender](#initializing-the-sender)
  - [Creating and Executing Jobs](#creating-and-executing-jobs)
- [Classes and Methods](#classes-and-methods)
  - [Sender Class](#sender-class)
    - [Initialization](#initialization)
    - [Connection Methods](#connection-methods)
    - [Job Control Methods](#job-control-methods)
    - [Laser Control Methods](#laser-control-methods)
    - [Utility Methods](#utility-methods)
  - [CommandList Class](#commandlist-class)
    - [Initialization](#initialization-1)
    - [Command Methods](#command-methods)
    - [Execution Methods](#execution-methods)
- [Examples](#examples)
  - [Simple Cutting Job](#simple-cutting-job)
  - [Handling Footswitch Events](#handling-footswitch-events)
- [Mock Connection for Testing](#mock-connection-for-testing)
- [Error Handling](#error-handling)
- [License](#license)

## Features

- Simplified control interface for BJJCZ laser control boards
- Blocking operations suitable for threaded applications
- Asynchronous job abortion capability
- Callback support for footswitch events
- Mock connection for testing without hardware
- Detailed control over laser parameters and movements

## Installation

This module requires Python 3.x and the following packages:

- `pyusb`: For USB communication with the laser control board
- `numpy`: For numerical operations within the `CommandList` class (if used)

Install the required packages using `pip`:

```bash
pip install pyusb numpy
```

## Getting Started

### Initializing the Sender

The `Sender` class is the primary interface for communicating with the laser control board.

```python
from sender import Sender

# Create a Sender instance
sender = Sender(debug=True)

# Open a connection to the laser control board
sender.open()
```

- The `debug` parameter enables verbose output for debugging purposes.
- The `open()` method establishes a connection to the first available laser control board.

### Creating and Executing Jobs

Use the `job()` method of the `Sender` class to create a `CommandList`, which represents a laser job.

```python
# Create a new CommandList (laser job)
job = sender.job()

# Add commands to the job
# ...

# Execute the job
job.execute()
```

## Classes and Methods

### Sender Class

The `Sender` class manages the communication with the laser control board, including opening and closing connections, sending commands, and handling job execution.

#### Initialization

```python
sender = Sender(footswitch_callback=None, debug=False)
```

- `footswitch_callback`: Optional. A function to be called when the footswitch is pressed.
- `debug`: Optional. Set to `True` to enable debug output.

#### Connection Methods

- `open(machine_index=0, mock=False, **kwargs)`: Opens a connection to the laser control board.
  - `machine_index`: Index of the machine to connect to (if multiple are available).
  - `mock`: If `True`, uses a mock connection for testing purposes.
  - `**kwargs`: Additional parameters for machine initialization.
- `close()`: Closes the connection to the laser control board.

#### Job Control Methods

- `job(*args, **kwargs)`: Creates a new `CommandList` associated with this `Sender`.
  - Returns a `CommandList` object to which you can add commands.
- `execute(command_list, loop_count=1, callback_finished=None)`: Executes a given `CommandList`.
  - `command_list`: The `CommandList` object containing the job commands.
  - `loop_count`: Number of times to repeat the job. Use `float('inf')` for infinite loops.
  - `callback_finished`: Optional. A function to be called when the job is finished.
- `abort()`: Aborts any running job.

#### Laser Control Methods

- `set_xy(x, y)`: Moves the laser head to the specified position.
- `get_xy()`: Retrieves the current position of the laser head.
- `light_on()`: Turns on the laser pointer or guide light (if available).
- `light_off()`: Turns off the laser pointer or guide light.

#### Utility Methods

- `is_ready()`: Checks if the laser control board is ready to receive commands.
- `is_busy()`: Checks if the laser control board is currently executing a job.
- `set_footswitch_callback(callback)`: Sets a callback function to be called when the footswitch is pressed.

### CommandList Class

The `CommandList` class represents a list of commands (operations) to be executed by the laser control board. You can add various operations to the command list to control the laser's behavior.

#### Initialization

```python
job = CommandList(sender=sender)
```

- `sender`: The `Sender` instance associated with this job.

#### Command Methods

- `append(operation)`: Adds a single operation to the command list.
- `extend(operations)`: Adds multiple operations to the command list.
- `clear()`: Clears all operations from the command list.
- `duplicate(begin, end, repeats=1)`: Duplicates a range of operations within the command list.
  - `begin`: Starting index.
  - `end`: Ending index.
  - `repeats`: Number of times to repeat the duplicated operations.

#### Execution Methods

- `execute(loop_count=1, *args, **kwargs)`: Executes the command list.
  - `loop_count`: Number of times to repeat the job.

## Examples

### Simple Cutting Job

```python
from sender import Sender
from command_list import CommandList, OpSetCutSpeed, OpSetLaserOnDelay, OpCut, OpLaserControl

# Initialize the sender
sender = Sender(debug=True)
sender.open()

# Create a new job
job = sender.job()

# Set laser parameters
job.append(OpSetCutSpeed(500))       # Set cutting speed to 500 units
job.append(OpSetLaserOnDelay(100))   # Set laser on delay to 100 microseconds
job.append(OpSetLaserOffDelay(100))  # Set laser off delay to 100 microseconds
job.append(OpLaserControl(1))        # Turn on the laser control

# Add cutting operations
job.append(OpCut(0x8100, 0x8200, 0, 0))  # Cut to position (0x8100, 0x8200)
job.append(OpCut(0x8300, 0x8400, 0, 0))  # Cut to position (0x8300, 0x8400)

# Execute the job
job.execute()

# Close the connection
sender.close()
```

### Handling Footswitch Events

```python
def footswitch_pressed(port_status):
    print("Footswitch pressed! Port status:", port_status)

# Initialize the sender with a footswitch callback
sender = Sender(footswitch_callback=footswitch_pressed, debug=True)
sender.open()

# Create and execute a job as before
# ...

# Close the connection
sender.close()
```

## Mock Connection for Testing

If you want to test your code without a physical laser control board connected, you can use the mock connection feature.

```python
sender = Sender(debug=True)
sender.open(mock=True)  # Use mock=True to enable mock connection

# Create and execute a job as before
# ...

sender.close()
```

With `mock=True`, the `Sender` uses the `MockConnection` class, which simulates communication with a laser control board.

## Error Handling

The module may raise the following exceptions:

- `BalorCommunicationException`: Raised when there is a communication error with the laser control board.
- `BalorMachineException`: Raised when the laser control board cannot be found or initialized.
- `BalorDataValidityException`: Raised when invalid data is sent to the laser control board.

Example of handling exceptions:

```python
try:
    sender.open()
except BalorMachineException as e:
    print("Error initializing the laser control board:", e)
```

## License

```
Balor Galvo Laser Control Module
Copyright (C) 2021-2022 Gnostic Instruments, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

For the full license text, see the [GNU General Public License](http://www.gnu.org/licenses/).

---

**Disclaimer:** Use this module responsibly and ensure compliance with all safety guidelines and regulations when operating laser equipment. Improper use can result in equipment damage or personal injury.