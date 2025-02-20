import Jetson.GPIO as GPIO
import time
import threading

class EncoderSpeed:
    def __init__(self, pin_a=11, pin_b=9, roller_pin=18):
        # Encoder pins and roller control pin
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.roller_pin = roller_pin

        # Encoder state variables
        self.counter = 0
        self.last_counter = 0
        self.rpm = 0
        self.pulses_per_rev = 400
        self.wheel_circumference = 0.1  # Meters, adjust to your setup

        # Initialize GPIO
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.roller_pin, GPIO.OUT, initial=GPIO.LOW)

        # Register encoder event detection
        GPIO.add_event_detect(self.pin_a, GPIO.BOTH, callback=self._encoder_callback)
        GPIO.add_event_detect(self.pin_b, GPIO.BOTH, callback=self._encoder_callback)

        # Thread lock and control flags
        self.lock = threading.Lock()
        self.stop_event = threading.Event()  # Graceful shutdown flag
        self.cutting_active = False  # Track cutting state

        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_speed, daemon=True)
        self.update_thread.start()

    def _encoder_callback(self, channel):
        """Handle encoder events."""
        a_state = GPIO.input(self.pin_a)
        b_state = GPIO.input(self.pin_b)

        with self.lock:
            if channel == self.pin_a:
                self.counter += 1 if a_state != b_state else -1
            else:
                self.counter += 1 if a_state == b_state else -1

    def _update_speed(self):
        """Continuously update speed."""
        while not self.stop_event.is_set():
            time.sleep(1)  # Update every second
            with self.lock:
                pulse_diff = abs(self.counter - self.last_counter)
                self.rpm = (pulse_diff / self.pulses_per_rev) * 60  # Calculate RPM
                self.last_counter = self.counter

    def get_speed(self):
        """Calculate and return speed in cm/min."""
        with self.lock:
            speed_mps = (self.rpm / 60) * self.wheel_circumference
            speed_cmpm = speed_mps * 6000 / 21.17  # Conversion factor
        return speed_cmpm

    def start_cutting(self):
        """Start cutting and activate the fabric roller."""
        with self.lock:
            self.cutting_active = True
            GPIO.output(self.roller_pin, GPIO.HIGH)  # Start roller
        print("Cutting started. Fabric roller activated.")

    def stop_cutting(self):
        """Stop cutting and deactivate the fabric roller."""
        with self.lock:
            self.cutting_active = False
            GPIO.output(self.roller_pin, GPIO.LOW)  # Stop roller
        print("Cutting stopped. Fabric roller deactivated.")

    def cleanup(self):
        """Clean up GPIO resources."""
        self.stop_event.set()  # Stop the update thread
        if GPIO.getmode() is not None:
            GPIO.remove_event_detect(self.pin_a)
            GPIO.remove_event_detect(self.pin_b)
            GPIO.cleanup()
        print("Resources cleaned up.")

# Usage example
if __name__ == "__main__":
    encoder = EncoderSpeed()

    try:
        while True:
            speed = encoder.get_speed()
            print(f"Fabric speed: {speed:.2f} cm/min")
            print(f"Counter: {encoder.counter}")

            # Example logic: Start cutting if speed exceeds 10 cm/min
            if speed > 10 and not encoder.cutting_active:
                encoder.start_cutting()
            elif speed <= 10 and encoder.cutting_active:
                encoder.stop_cutting()

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting...")
        encoder.cleanup()
    except Exception as e:
        print(f"An error occurred: {e}")
        encoder.cleanup()
