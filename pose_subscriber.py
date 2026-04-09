#!/usr/bin/env python3
"""
Subscribe to FoundationPose poses over ZMQ.

Run on the robot:
    python3 pose_subscriber.py --host 192.168.123.162

The pose arrives as a dict with:
    - timestamp: float (unix time)
    - frame: int
    - position: {x, y, z} in meters
    - orientation: {x, y, z, w} quaternion

You can import PoseSubscriber into your own code:
    from pose_subscriber import PoseSubscriber
    sub = PoseSubscriber("192.168.123.162")
    while True:
        pose = sub.get_latest()
        if pose is not None:
            print(pose["position"])
"""

import argparse
import json
import time

import zmq


class PoseSubscriber:
    def __init__(self, host="localhost", port=5555):
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{host}:{port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
        self.latest = None

    def get_latest(self, timeout_ms=100):
        """Get the latest pose. Returns None if no message within timeout."""
        if self.sub.poll(timeout_ms):
            self.latest = self.sub.recv_json()
        return self.latest

    def close(self):
        self.sub.close()
        self.ctx.term()


def main():
    parser = argparse.ArgumentParser(description="FoundationPose subscriber")
    parser.add_argument('--host', type=str, default='192.168.123.162',
                        help='IP of the machine running run_realsense.py')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    sub = PoseSubscriber(args.host, args.port)
    print(f"Subscribing to tcp://{args.host}:{args.port}")
    print("Waiting for poses...\n")

    try:
        count = 0
        while True:
            pose = sub.get_latest()
            if pose is not None:
                count += 1
                p = pose["position"]
                o = pose["orientation"]
                if count % 10 == 0:  # Print every 10th pose
                    print(f"[{count:>5}] "
                          f"pos=({p['x']:+.4f}, {p['y']:+.4f}, {p['z']:+.4f})  "
                          f"quat=({o['x']:+.4f}, {o['y']:+.4f}, {o['z']:+.4f}, {o['w']:+.4f})")
    except KeyboardInterrupt:
        print(f"\nReceived {count} poses total.")
    finally:
        sub.close()


if __name__ == '__main__':
    main()
