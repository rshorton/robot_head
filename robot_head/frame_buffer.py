from collections import deque
import threading
from rclpy.time import Time, Duration
import rclpy.logging

class ExpiringFrameBuffer:
    def __init__(self, holding_period_seconds: float):
        self.holding_period = Duration(seconds=holding_period_seconds)
        # Store tuples of (rclpy.time.Time, frame_data)
        self.buffer = deque()
        self.lock = threading.Lock()
        self.logger = rclpy.logging.get_logger('expiring_frame_buffer')

    def purge_expired(self, current_time: Time):
        """Removes frames that have exceeded the holding period."""
        # Deque is ordered, so we only need to check and pop from the left
        while self.buffer and (current_time - self.buffer[0][0]) > self.holding_period:
            seconds, nanoseconds = self.buffer[0][0].seconds_nanoseconds()
            self.logger.info(f"Purging frame: {seconds}.{nanoseconds}")
            self.buffer.popleft()

    def add_frame(self, msg_time: Time, frame_depth, frame_color):
        """
        Adds a frame indexed by its ROS 2 Time.
        msg_time should ideally come from msg.header.stamp.
        """
        with self.lock:
            self.purge_expired(msg_time)
            self.buffer.append((msg_time, frame_depth, frame_color))

            seconds, nanoseconds = msg_time.seconds_nanoseconds()
            self.logger.info(f"Adding frame: {seconds}.{nanoseconds}")

    def lookup_frame(self, target_time: Time, max_tolerance_seconds: float = 0.05):
        """
        Finds the closest frame to the target_time within a maximum tolerance window.
        Returns (frame, actual_time) if found, or (None, None).
        """
        with self.lock:
            if not self.buffer:
                return None, None, None
                
            tolerance = Duration(seconds=max_tolerance_seconds)
            best_frame = None
            best_time = None
            smallest_diff = None

            for msg_time, frame_depth, frame_color in self.buffer:
                # Calculate absolute time difference
                self.logger.debug(f"buffer time: {msg_time}")
                self.logger.debug(f"target_time time: {target_time}")

                diff = abs((target_time - msg_time).nanoseconds)
                if smallest_diff is None or diff < smallest_diff:
                    smallest_diff = diff
                    best_frame_depth = frame_depth
                    best_frame_color = frame_color
                    best_time = msg_time
            
            # Check if the closest match falls within the allowed tolerance
            if smallest_diff is not None and Duration(nanoseconds=smallest_diff) <= tolerance:
                return best_frame_depth, best_frame_color, best_time
                
            return None, None, None