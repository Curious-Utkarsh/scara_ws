#!/usr/bin/env python3
"""Pick the box selected by the fixed-camera colour detector."""

import math

import rclpy
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener

from pymoveit2 import GripperInterface, MoveIt2
from pymoveit2.robots import scara


# These are the values used by camera_fixed in scara_description/urdf/gazebo.xacro.
HORIZONTAL_FOV = 1.3962634

# Transient-local + depth 1: the last colour command is latched, so this node
# gets it on subscribe even if fixed_camera_color_detector published first.
PICK_COMMAND_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)

# shoulder_joint's lower limit (scara_description/urdf/ros2control.xacro): fully lowered.
GRASP_SHOULDER_POSITION = -0.15
GRASP_JOINT_INDEX = scara.joint_names().index("shoulder_joint")

# Drop joint targets, matching the drop_approach/drop_release/drop_home entries in
# pymoveit2/examples/pick_and_drop.py -- the pick side there is now handled by vision.
DROP_TARGETS = {
    "R": {
        "drop_approach": [-1.611, 0.0, -2.00, 0.0],
        "drop_release": [-1.611, -0.05, -2.00, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
    "G": {
        "drop_approach": [-1.611, 0.0, 2.00, -0.184],
        "drop_release": [-1.611, 0.0, 0.0, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
    "B": {
        "drop_approach": [1.611, 0.0, -2.00, 0.184],
        "drop_release": [1.611, 0.0, 0.0, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
}


class VisionPickAndPlace(Node):
    def __init__(self):
        super().__init__("vision_pick_and_place")
        # "fixed" lowers shoulder_joint to its mechanical floor to grasp, since a
        # single top-down depth reading only gives the box's top surface, not its
        # height. "camera" descends to that detected surface instead -- useful if
        # boxes vary in height and the fixed floor would be too high to reach them.
        self.declare_parameter("grasp_z_source", "fixed")
        self.grasp_z_source = str(self.get_parameter("grasp_z_source").value).lower()
        if self.grasp_z_source not in ("fixed", "camera"):
            raise ValueError("grasp_z_source must be 'fixed' or 'camera'")

        self.bridge = CvBridge()
        self.depth_image = None
        self.busy = False
        self.target_color = None
        self.pending_pick = None

        group = ReentrantCallbackGroup()
        self.arm = MoveIt2(
            node=self,
            joint_names=scara.joint_names(),
            base_link_name=scara.base_link_name(),
            end_effector_name=scara.end_effector_name(),
            group_name=scara.MOVE_GROUP_ARM,
            callback_group=group,
            use_move_group_action=True,
        )
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=scara.gripper_joint_names(),
            open_gripper_joint_positions=scara.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=scara.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=scara.MOVE_GROUP_GRIPPER,
            callback_group=group,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, "/camera_fixed/depth_image", self.depth_callback, 10)
        self.create_subscription(PointStamped, "/scara/pick_target_pixel", self.target_callback, 10)
        # Transient-local so the detector's latched colour is received even if it
        # published before this node subscribed (start order shouldn't matter).
        self.create_subscription(
            String, "/scara/pick_command", self.command_callback, PICK_COMMAND_QOS
        )
        self.get_logger().info("Waiting for a colour command on /scara/pick_command.")

    def command_callback(self, msg):
        color = msg.data.strip().upper()
        if color not in DROP_TARGETS:
            self.get_logger().warning(f"Ignoring command '{msg.data}'; expected R, G, or B.")
            return
        if self.target_color != color:
            self.target_color = color
            self.get_logger().info(f"Now looking for the {color} box.")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def target_callback(self, pixel):
        if self.busy or self.target_color is None or self.depth_image is None:
            return

        # Read the real RGB-D depth at the detected box centre pixel.
        u, v = int(pixel.point.x), int(pixel.point.y)
        if u < 0 or v < 0 or v >= self.depth_image.shape[0] or u >= self.depth_image.shape[1]:
            return
        depth = float(self.depth_image[v, u])
        if self.depth_image.dtype == np.uint16:
            depth /= 1000.0
        if not np.isfinite(depth) or depth <= 0.0:
            return

        # Calculate camera intrinsics from the fixed camera's configured FOV.
        height, width = self.depth_image.shape[:2]
        fx = width / (2.0 * math.tan(HORIZONTAL_FOV / 2.0))
        fy = fx
        cx, cy = width / 2.0, height / 2.0

        # Convert the pixel (u, v) and its real depth into camera-frame metres.
        camera_point = PointStamped()
        camera_point.header.stamp = pixel.header.stamp
        camera_point.header.frame_id = pixel.header.frame_id
        camera_point.point.x = (u - cx) * depth / fx
        camera_point.point.y = (v - cy) * depth / fy
        camera_point.point.z = depth

        try:
            camera_to_base = self.tf_buffer.lookup_transform(
                "base_link", camera_point.header.frame_id, Time()
            )
            box = do_transform_point(camera_point, camera_to_base)
        except TransformException as error:
            self.get_logger().warning(f"No camera-to-base TF yet: {error}")
            return

        self.get_logger().info(
            f"Pixel=({u}, {v}) depth={depth:.3f} m | "
            f"camera=({camera_point.point.x:.3f}, {camera_point.point.y:.3f}, "
            f"{camera_point.point.z:.3f}) | base=({box.point.x:.3f}, "
            f"{box.point.y:.3f}, {box.point.z:.3f})"
        )
        # Defer the actual motion to the main loop instead of spinning it off
        # onto a separate thread: MoveIt2's blocking calls (wait_until_executed,
        # compute_ik) do their own internal rclpy.spin_once on this node, and
        # running that from a second thread while this node is also being spun
        # here would race two spinners over the same subscriptions -- messages
        # (like a colour command) can then go missing with no error at all.
        self.busy = True
        self.pending_pick = (box, self.target_color)

    def solve_ik(self, position):
        # OMPL's Cartesian pose-goal sampling is unreliable near this SCARA's
        # reach limit (RRTConnect can't randomly sample a goal state that is
        # close to a singularity). Solve IK once from the current joint state
        # instead, then send a joint-space goal like the other examples do.
        ik_result = self.arm.compute_ik(
            position=position,
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        if ik_result is None:
            return None
        return [
            ik_result.position[ik_result.name.index(name)]
            for name in scara.joint_names()
        ]

    def move_to(self, joint_positions):
        self.arm.move_to_configuration(joint_positions, joint_names=scara.joint_names())
        return self.arm.wait_until_executed()

    def pick_box(self, box, color):
        x, y, z = box.point.x, box.point.y, box.point.z
        try:
            self.gripper.open()
            if not self.gripper.wait_until_executed():
                return

            # Move above the box. Its XY and approach height come from vision.
            approach = self.solve_ik([x, y, z + 0.10])
            if approach is None or not self.move_to(approach):
                self.get_logger().error(f"No reachable approach pose above {(x, y, z)}")
                return

            if self.grasp_z_source == "camera":
                grasp = self.solve_ik([x, y, z])  # descend to the detected surface
                if grasp is None:
                    self.get_logger().error(f"No reachable grasp pose at {(x, y, z)}")
                    return
            else:
                # Grasp at the mechanical floor instead of the detected Z --
                # the same fixed height every colour uses in pick_and_place.py
                # / pick_and_drop.py.
                grasp = list(approach)
                grasp[GRASP_JOINT_INDEX] = GRASP_SHOULDER_POSITION
            if not self.move_to(grasp):
                return

            self.gripper.close()
            if not self.gripper.wait_until_executed():
                return

            if not self.move_to(approach):  # lift the box back up
                return

            # Drop it at this colour's calibrated location, like pick_and_drop.py.
            drop = DROP_TARGETS[color]
            if not self.move_to(drop["drop_approach"]):
                return
            if not self.move_to(drop["drop_release"]):
                return
            self.gripper.open()
            if not self.gripper.wait_until_executed():
                return
            self.move_to(drop["drop_home"])
            self.get_logger().info(f"{color} pick-and-drop complete.")
        except Exception as error:
            self.get_logger().error(f"Pick failed: {error}")
        finally:
            self.busy = False


def main():
    rclpy.init()
    node = VisionPickAndPlace()
    try:
        while rclpy.ok():
            # This is the only spinner. pick_box's blocking calls also spin
            # this same node, so topic callbacks received during a pick queue
            # up and are handled as soon as pick_box returns.
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.pending_pick is not None:
                box, color = node.pending_pick
                node.pending_pick = None
                node.pick_box(box, color)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
