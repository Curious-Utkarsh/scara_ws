#!/usr/bin/env python3
"""Pick one colour-box target published by scara_vision/color_box_detector."""

from threading import Thread

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pymoveit2 import GripperInterface, MoveIt2
from pymoveit2.robots import scara


class VisionPickAndPlace(Node):
    def __init__(self):
        super().__init__("vision_pick_and_place")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("approach_height", 0.10)
        self.declare_parameter("grasp_z_offset", -0.02)
        self.declare_parameter("lift_height", 0.12)
        self.declare_parameter("drop_position", [0.30, 0.25, 0.40])
        self.declare_parameter("tool_quat_xyzw", [0.0, 0.0, 0.0, 1.0])

        callback_group = ReentrantCallbackGroup()
        self.arm = MoveIt2(
            node=self, joint_names=scara.joint_names(),
            base_link_name=scara.base_link_name(), end_effector_name=scara.end_effector_name(),
            group_name=scara.MOVE_GROUP_ARM, callback_group=callback_group,
        )
        self.gripper = GripperInterface(
            node=self, gripper_joint_names=scara.gripper_joint_names(),
            open_gripper_joint_positions=scara.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=scara.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=scara.MOVE_GROUP_GRIPPER, callback_group=callback_group,
        )
        self.busy = False
        colour = str(self.get_parameter("target_color").value).lower()
        if colour not in ("red", "green", "blue"):
            raise ValueError("target_color must be red, green, or blue")
        self.create_subscription(PoseStamped, f"/scara/detections/{colour}", self.target_callback, 10)
        self.get_logger().info(f"Waiting for one {colour} target from the vision node.")

    def target_callback(self, target):
        if self.busy:
            return
        self.busy = True  # Pick one target only; restart the node for the next box.
        Thread(target=self.pick_and_place, args=(target,), daemon=True).start()

    def move(self, label, position):
        self.get_logger().info(label)
        self.arm.move_to_pose(
            position=position, quat_xyzw=self.get_parameter("tool_quat_xyzw").value,
            frame_id="base_link", tolerance_position=0.01, tolerance_orientation=0.05,
        )
        return self.arm.wait_until_executed()

    def set_gripper(self, close):
        self.gripper.close() if close else self.gripper.open()
        return self.gripper.wait_until_executed()

    def pick_and_place(self, target):
        x, y, z = target.pose.position.x, target.pose.position.y, target.pose.position.z
        approach = self.get_parameter("approach_height").value
        grasp_z = z + self.get_parameter("grasp_z_offset").value
        lift_z = z + self.get_parameter("lift_height").value
        drop = list(self.get_parameter("drop_position").value)
        try:
            if not self.set_gripper(False):
                return
            if not self.move("Move above box", [x, y, z + approach]):
                return
            if not self.move("Lower to box", [x, y, grasp_z]):
                return
            if not self.set_gripper(True):
                return
            if not self.move("Lift box", [x, y, lift_z]):
                return
            if not self.move("Move above drop point", [drop[0], drop[1], drop[2] + approach]):
                return
            if not self.move("Lower at drop point", drop):
                return
            self.set_gripper(False)
            self.move("Retreat", [drop[0], drop[1], drop[2] + approach])
        except Exception as error:
            self.get_logger().error(f"Pick-and-place failed: {error}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionPickAndPlace()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
