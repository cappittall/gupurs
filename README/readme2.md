# User Manual: Managing a Galvo Laser Cutter with Balor Control Module

## 1. Introduction

This manual provides instructions for managing a galvo laser cutter using the Balor Galvo Laser Control Module. The module interfaces with BJJCZ (Golden Orange, Beijing JCZ) LMCV4-FIBER-M and compatible boards.

## 2. Setup and Initialization

### 2.1 Importing the Module

```python
from balor.sender import Sender

2.2 Creating a Sender Instance
sender = Sender(footswitch_callback=None, debug=False)

footswitch_callback: Optional function to handle footswitch events
debug: Set to True for detailed logging
2.3 Opening a Connection
sender.open(machine_index=0, mock=False)

machine_index: Use 0 for the first connected device
mock: Set to True for testing without hardware
3. Basic Operations
3.1 Creating a Job
job = sender.job()

3.2 Executing a Job
sender.execute(job, loop_count=1, callback_finished=None)

loop_count: Number of times to repeat the job (use float('inf') for infinite loops)
callback_finished: Optional function to call when job completes
3.3 Aborting a Job
sender.abort()

3.4 Checking Machine Status
is_ready = sender.is_ready()
is_busy = sender.is_busy()

4. Laser Control
4.1 Enabling/Disabling Laser
sender.raw_enable_laser()
sender.raw_disable_laser()

4.2 Laser Signal Control
sender.raw_laser_signal_on()
sender.raw_laser_signal_off()

5. Galvo Control
5.1 Setting Galvo Position
sender.set_xy(x, y)

5.2 Getting Galvo Position
x, y = sender.get_xy()

6. Machine Configuration
6.1 Setting Control Mode
sender.raw_set_control_mode(mode, value)

6.2 Setting Laser Mode
sender.raw_set_laser_mode(mode, value)

6.3 Setting Timing and Delays
sender.raw_set_timing(mode, value)
sender.raw_set_delay_mode(mode, value)

7. Advanced Features
7.1 Loading Correction Table
sender._read_correction_file(filename)
sender._send_correction_table(table)

7.2 First Pulse Killer Settings
sender.raw_set_first_pulse_killer(value, stack)
sender.raw_set_fpk_param_2(v1, v2, v3, v4)

7.3 PWM Control
sender.raw_set_pwm_half_period(value, stack)
sender.raw_set_pwm_pulse_width(value, stack)

8. I/O Operations
8.1 Port Control
sender.port_on(bit)
sender.port_off(bit)
sender.port_toggle(bit)

8.2 Reading Port Status
port_status = sender.read_port()

8.3 Analog Port Control
sender.raw_write_analog_port_1(value, stack)
sender.raw_write_analog_port_2(value, stack)

9. Maintenance and Diagnostics
9.1 Getting Serial Number and Version
serial_number = sender.raw_get_serial_no()
version = sender.raw_get_version()

9.2 Resetting the Machine
sender.raw_reset()

10. Closing the Connection
sender.close()

11. Error Handling
The module may raise the following exceptions:

BalorException: Base exception for all Balor-related errors
BalorMachineException: Issues with the laser cutter hardware
BalorCommunicationException: Communication errors with the device
BalorDataValidityException: Invalid data or parameters
Always wrap operations in try-except blocks to handle these exceptions gracefully.

12. Safety Precautions
Always wear appropriate eye protection when operating the laser cutter.
Ensure proper ventilation in the work area.
Never leave the laser cutter unattended while in operation.
Regularly inspect and maintain the machine according to manufacturer guidelines.

This user manual provides a comprehensive overview of managing a galvo laser cutter using the Balor Control Module. It covers setup, basic operations, laser and galvo control, machine configuration, advanced features, I/O operations, maintenance, and safety precautions. Users should refer to this guide alongside the specific documentation for their laser cutter model for optimal and safe operation.



# CommandList Class Documentation

The `CommandList` class is a powerful tool for generating and managing laser cutting/engraving commands. It provides a high-level interface to create and manipulate laser operations.

## Basic Usage

```python
from commandlist import CommandList

# Create a new CommandList instance
job = CommandList()

# Set initial position
job.init(0, 0)

# Configure laser settings
job.set_mark_settings(
    travel_speed=100,
    frequency=20,
    power=50,
    cut_speed=10,
    laser_on_delay=100,
    laser_off_delay=100,
    polygon_delay=10
)

# Add operations
job.goto(10, 10)
job.mark(20, 20)
job.light(30, 30, light=True)

# Generate binary data
binary_data = job.serialize()



Key Methods
Initialization and Configuration
init(x, y): Set the initial position
set_mark_settings(...): Configure multiple laser settings at once
set_travel_speed(speed): Set travel speed
set_cut_speed(speed): Set cutting speed
set_power(power): Set laser power (0-100%)
set_frequency(frequency): Set Q-switch frequency (kHz)
Movement and Marking
goto(x, y): Move to a position without firing the laser
mark(x, y): Move to a position while firing the laser
light(x, y, light=True): Move to a position with light on/off
Laser Control
laser_control(control): Enable/disable laser control
light_on(): Turn on the light
light_off(): Turn off the light
Delays and Timing
set_laser_on_delay(delay): Set laser-on delay
set_laser_off_delay(delay): Set laser-off delay
set_polygon_delay(delay): Set polygon delay
set_mark_end_delay(delay): Set mark end delay
jump_delay(delay): Set jump delay
Data Generation
serialize(): Generate binary data for the entire job
packet_generator(): Generate binary data in packets
Debugging
plot(draw, resolution=2048, show_travels=False): Visualize the job
serialize_to_file(file): Save binary data to a file
Advanced Usage
The CommandList class also provides low-level "raw" methods for direct command insertion, as well as methods for port manipulation and other specialized operations. Refer to the class implementation for details on these advanced features.


This documentation provides an overview of the main features and usage of the CommandList class. You may want to expand on certain sections or add more examples depending on your specific needs and target audience.




