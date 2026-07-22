#!/usr/bin/env python3
"""Simple red, green and blue box detector for the simulated RGB-D camera."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


# HSV colour ranges: red needs two ranges because HSV hue wraps from 180 to 0.
RANGES = {
    "Red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (180, 255, 255))],
    "Green": [((40, 70, 50), (85, 255, 255))],
    "Blue": [((100, 100, 50), (130, 255, 255))],
}
DRAW = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}


class ColorBoxDetector(Node):
    def __init__(self):
        super().__init__("color_box_detector")
        self.bridge = CvBridge()
        self.depth = None
        self.create_subscription(Image, "/camera/depth_image", self.depth_cb, 10)
        self.create_subscription(Image, "/camera/image", self.image_cb, 10)
        cv2.namedWindow("Colour Box Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Colour Box Detector", 1280, 960)

    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def image_cb(self, msg):
        if self.depth is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for name, ranges in RANGES.items():
            mask = sum((cv2.inRange(hsv, np.array(low), np.array(high))
                        for low, high in ranges), np.zeros(hsv.shape[:2], np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 3000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            depth_m = float(self.depth[cy, cx])
            if self.depth.dtype == np.uint16:
                depth_m /= 1000.0

            colour = DRAW[name]
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
            cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(image, f"Detected: {name} Box  {depth_m:.3f} m",
                        (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, colour, 2)

        cv2.imshow("Colour Box Detector", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ColorBoxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
