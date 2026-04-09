#!/usr/bin/env python3
"""
ROS2 bridge: subscribes to ZMQ pose messages from run_realsense.py and
publishes geometry_msgs/PoseStamped on /foundationpose/pose.

Run on the robot:
    source ~/ros2_humble/install/setup.bash
    python3 ~/ros2_pose_bridge.py

Args:
    --zmq_host: IP of the machine running run_realsense.py (default: 192.168.123.162)
    --zmq_port: ZMQ port (default: 5555)
    --topic: ROS2 topic name (default: /foundationpose/pose)
    --frame_id: TF frame ID (default: camera_color_optical_frame)
"""

import argparse
import zmq
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class PoseBridge(Node):
    def __init__(self, topic, frame_id, zmq_host, zmq_port, zmq_bind):
        super().__init__('foundationpose_bridge')
        self.pub = self.create_publisher(PoseStamped, topic, 10)
        self.frame_id = frame_id

        # ZMQ subscriber
        self.zmq_ctx = zmq.Context()
        self.zmq_sub = self.zmq_ctx.socket(zmq.SUB)
        if zmq_bind:
            self.zmq_sub.bind("tcp://*:{}".format(zmq_port))
            self.get_logger().info(
                "ZMQ binding on tcp://*:{} -> {}".format(zmq_port, topic))
        else:
            self.zmq_sub.connect("tcp://{}:{}".format(zmq_host, zmq_port))
            self.get_logger().info(
                "Bridging tcp://{}:{} -> {}".format(zmq_host, zmq_port, topic))
        self.zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_sub.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message

        # Poll at 100Hz
        self.timer = self.create_timer(0.01, self.poll_zmq)
        self.msg_count = 0

    def poll_zmq(self):
        try:
            data = self.zmq_sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = data["position"]["x"]
        msg.pose.position.y = data["position"]["y"]
        msg.pose.position.z = data["position"]["z"]
        msg.pose.orientation.x = data["orientation"]["x"]
        msg.pose.orientation.y = data["orientation"]["y"]
        msg.pose.orientation.z = data["orientation"]["z"]
        msg.pose.orientation.w = data["orientation"]["w"]

        self.pub.publish(msg)
        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info("Published {} poses".format(self.msg_count))

    def destroy_node(self):
        self.zmq_sub.close()
        self.zmq_ctx.term()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="ZMQ-to-ROS2 pose bridge")
    parser.add_argument('--zmq_host', type=str, default='localhost')
    parser.add_argument('--zmq_port', type=int, default=5555)
    parser.add_argument('--zmq_bind', action='store_true',
                        help='Bind ZMQ socket instead of connecting (use on robot)')
    parser.add_argument('--topic', type=str, default='/foundationpose/pose')
    parser.add_argument('--frame_id', type=str,
                        default='camera_color_optical_frame')
    args = parser.parse_args()

    rclpy.init()
    node = PoseBridge(args.topic, args.frame_id, args.zmq_host, args.zmq_port, args.zmq_bind)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
