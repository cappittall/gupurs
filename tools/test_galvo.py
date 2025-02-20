import usb.core
import usb.util

# Find the device
dev = usb.core.find(idVendor=0x9588, idProduct=0x9899)

# Was it found?
if dev is None:
    raise ValueError('Device not found')

# Set the active configuration. With no arguments, the first
# configuration will be the active one
dev.set_configuration()

# Get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(0,0)]

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)

assert ep is not None

# Write a test message
ep.write('Hello, Galvo!')

print('Message sent to device')
