#!/usr/bin/env python3
"""
camera_node.py
Streams the Raspberry Pi Camera Module 3 into ROS2 using rpicam-vid (MJPEG)
and publishes each frame as sensor_msgs/CompressedImage.

Topic published:  /camera/image/compressed
"""
import ctypes
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

def _die_with_parent():
    # Linux: child gets SIGTERM if the camera node dies (no orphan rpicam-vid)
    try:
        ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa
        pass


SOI = b'\xff\xd8'  # JPEG start
EOI = b'\xff\xd9'  # JPEG end


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 10)
        self.declare_parameter('quality', 80)
        self.declare_parameter('topic', '/camera/image/compressed')
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('camera_cmd', 'rpicam-vid')
        self.declare_parameter('lens_position', -1.0)   # -1 = default autofocus; else fixed focus in dioptres (1/distance_m, e.g. 4.0 = 25 cm)
        self.declare_parameter('extra_args', '')        # e.g. '--ev 0.5 --sharpness 1.5'

        self.w = int(self.get_parameter('width').value)
        self.h = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.quality = int(self.get_parameter('quality').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.cmd = self.get_parameter('camera_cmd').value
        self.lens_position = float(self.get_parameter('lens_position').value)
        self.extra_args = str(self.get_parameter('extra_args').value).split()
        topic = self.get_parameter('topic').value

        self.pub = self.create_publisher(
            CompressedImage, topic, qos_profile_sensor_data)

        self.proc = None
        self.running = True
        self.frames = 0
        self._last_count = 0
        self.create_timer(5.0, self._report)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        self.get_logger().info(
            f'Camera node started ({self.w}x{self.h} @ {self.fps} fps) -> {topic}')

    def _report(self):
        n = self.frames - self._last_count
        self._last_count = self.frames
        if n == 0:
            self.get_logger().warn(
                'NO frames in the last 5 s - check camera cable / other program using it '
                '(test: rpicam-hello -t 3000)')
        else:
            self.get_logger().info(f'Camera OK: {n / 5.0:.1f} fps')

    # ------------------------------------------------------------------
    def _start_proc(self):
        cmd = [
            self.cmd, '-t', '0', '-n',
            '--codec', 'mjpeg',
            '--width', str(self.w), '--height', str(self.h),
            '--framerate', str(self.fps),
            '--quality', str(self.quality),
            '-o', '-',
        ]
        if self.lens_position >= 0:
            cmd += ['--autofocus-mode', 'manual', '--lens-position', str(self.lens_position)]
        cmd += self.extra_args
        # free the camera if an old rpicam process is still holding it
        subprocess.run(['pkill', '-x', self.cmd], check=False)
        time.sleep(0.5)
        self.get_logger().info('Starting: ' + ' '.join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=None, bufsize=0,
            preexec_fn=_die_with_parent)  # stderr visible for debugging

    def _reader(self):
        buf = b''
        while self.running:
            if self.proc is None or self.proc.poll() is not None:
                try:
                    self._start_proc()
                    buf = b''
                except Exception as e:  # noqa
                    self.get_logger().error(f'Cannot start camera: {e}')
                    time.sleep(2.0)
                    continue

            chunk = self.proc.stdout.read(65536)
            if not chunk:
                self.get_logger().warn('Camera stream ended, restarting...')
                time.sleep(1.0)
                self.proc = None
                continue
            buf += chunk

            # extract every complete JPEG in the buffer
            while True:
                a = buf.find(SOI)
                if a == -1:
                    buf = b''
                    break
                b = buf.find(EOI, a + 2)
                if b == -1:
                    buf = buf[a:]
                    break
                jpg = buf[a:b + 2]
                buf = buf[b + 2:]
                self._publish(jpg)

            if len(buf) > 5_000_000:  # safety against garbage
                buf = b''

    def _publish(self, jpg: bytes):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = jpg
        self.pub.publish(msg)
        self.frames += 1

    def destroy_node(self):
        self.running = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:  # noqa
                self.proc.kill()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
