#!/usr/bin/env python3
"""Quick test: publish a dummy PoseStamped on /foundationpose/pose at 10Hz."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math


class TestPub(Node):
    def __init__(self):
        super().__init__('test_pose_pub')
        self.pub = self.create_publisher(PoseStamped, '/foundationpose/pose', 10)
        self.timer = self.create_timer(0.1, self.publish)
        self.count = 0
        self.get_logger().info('Publishing dummy poses on /foundationpose/pose')

    def publish(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.pose.position.x = math.sin(self.count * 0.1) * 0.1
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.5
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)
        self.count += 1
        if self.count % 50 == 0:
            self.get_logger().info('Published {} poses'.format(self.count))


def main():
    rclpy.init()
    node = TestPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
