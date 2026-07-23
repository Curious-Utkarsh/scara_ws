#!/usr/bin/env python3
"""Find one coloured box in the fixed camera image and publish its pixel centre."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


# HSV ranges for red, green and blue. Red needs two ranges in HSV.
COLOURS = {
    "r": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (180, 255, 255))],
    "g": [((40, 70, 50), (85, 255, 255))],
    "b": [((100, 100, 50), (130, 255, 255))],
}

# Transient-local + depth 1: latches the colour command so vision_pick_and_place
# gets it on subscribe even if this node published before that one started.
PICK_COMMAND_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


class FixedCameraColorDetector(Node):
    def __init__(self):
        super().__init__("fixed_camera_color_detector")

        # r = red, g = green, b = blue. Default is red.
        self.declare_parameter("pick", "r")
        self.pick = str(self.get_parameter("pick").value).lower()[0]
        if self.pick not in COLOURS:
            raise ValueError("pick must be r, g, or b")

        self.bridge = CvBridge()
        self.depth_image = None
        self.publisher = self.create_publisher(
            PointStamped, "/scara/pick_target_pixel", 10
        )
        # Tells vision_pick_and_place which colour to pick, so it needs no
        # parameter of its own. Latched (transient local) so it's still there
        # for that node even if it subscribes after this message is sent.
        self.command_publisher = self.create_publisher(
            String, "/scara/pick_command", PICK_COMMAND_QOS
        )
        self.create_subscription(Image, "/camera_fixed/image", self.image_callback, 10)
        self.create_subscription(Image, "/camera_fixed/depth_image", self.depth_callback, 10)
        self.create_subscription(String, "/scara/pick_command", self.command_callback, PICK_COMMAND_QOS)
        cv2.namedWindow("Fixed Camera Box Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fixed Camera Box Detector", 1280, 960)
        self.get_logger().info(f"Looking for {self.pick} box")
        self.command_publisher.publish(String(data=self.pick.upper()))

    def command_callback(self, msg):
        color = msg.data.strip().lower()[:1]
        if color not in COLOURS:
            self.get_logger().warning(f"Ignoring command '{msg.data}'; expected R, G, or B.")
            return
        self.pick = color
        self.get_logger().info(f"Now looking for {self.pick} box")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in COLOURS[self.pick]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            box = max(contours, key=cv2.contourArea)
            if cv2.contourArea(box) >= 3000:
                x, y, width, height = cv2.boundingRect(box)
                target = PointStamped()
                target.header = msg.header
                target.point.x = float(x + width // 2)  # image x pixel
                target.point.y = float(y + height // 2)  # image y pixel
                target.point.z = 0.0                    # no depth is used
                self.publisher.publish(target)

                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.circle(image, (int(target.point.x), int(target.point.y)), 5, (255, 255, 255), -1)
                if self.depth_image is not None:
                    depth = float(self.depth_image[int(target.point.y), int(target.point.x)])
                    if self.depth_image.dtype == np.uint16:
                        depth /= 1000.0
                    cv2.putText(image, f"pixel=({int(target.point.x)}, {int(target.point.y)}) depth={depth:.3f} m",
                                (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2)

        cv2.imshow("Fixed Camera Box Detector", image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = FixedCameraColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
