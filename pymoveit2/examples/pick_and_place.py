#!/usr/bin/env python3
"""Joint-space pick-and-place routine for the custom SCARA robot."""

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pymoveit2 import GripperInterface, MoveIt2
from pymoveit2.robots import scara


# Joint order: column_joint, shoulder_joint, forearm_joint, wrist_joint.
# Add the calibrated Red and Blue sequences here when those locations are known.
TARGETS = {
    "R": {
        "home": [0.0, 0.0, 0.0, 0.0],
        "pick_approach": [0.0, -0.15, 0.0, 0.0],
        "pick_grasp": [0.0, -0.15, 0.0, 0.0],
        "pick_lift": [0.0, 0.0, 0.0, 0.0],
        "drop_approach": [-1.611, 0.0, -2.00, 0.0],
        "drop_release": [-1.611, 0.0, -2.00, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
    "G": {
        "home": [0.0, 0.0, 0.0, 0.0],
        "pick_approach": [-1.438, 0.0, 2.00, -0.184],
        "pick_grasp": [-1.438, -0.15, 2.00, -0.184],
        "pick_lift": [-1.438, 0.0, 2.00, -0.184],
        "drop_approach": [-1.611, 0.0, 2.00, -0.184],
        "drop_release": [-1.611, 0.0, 0.0, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
    "B": {
        "home": [0.0, 0.0, 0.0, 0.0],
        "pick_approach": [1.438, 0.0, -2.00, 0.184],
        "pick_grasp": [1.438, -0.15, -2.00, 0.184],
        "pick_lift": [1.438, 0.0, -2.00, 0.184],
        "drop_approach": [1.611, 0.0, -2.00, 0.184],
        "drop_release": [1.611, 0.0, 0.0, 0.0],
        "drop_home": [0.0, 0.0, 0.0, 0.0],
    },
}


class PickAndPlace(Node):
    """Run one calibrated pick-and-place sequence."""

    def __init__(self):
        super().__init__("pick_and_place")
        self.declare_parameter("target_color", "G")

        callback_group = ReentrantCallbackGroup()
        self.arm = MoveIt2(
            node=self,
            joint_names=scara.joint_names(),
            base_link_name=scara.base_link_name(),
            end_effector_name=scara.end_effector_name(),
            group_name=scara.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=scara.gripper_joint_names(),
            open_gripper_joint_positions=scara.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=scara.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=scara.MOVE_GROUP_GRIPPER,
            callback_group=callback_group,
        )

    def move(self, step, joints):
        """Send one joint-space target and wait for it to finish."""
        self.get_logger().info(step)
        self.arm.move_to_configuration(joints)
        if not self.arm.wait_until_executed():
            self.get_logger().error(f"Failed: {step}")
            return False
        return True

    def set_gripper(self, step, close):
        """Open or close the gripper and wait for completion."""
        self.get_logger().info(step)
        if close:
            self.gripper.close()
        else:
            self.gripper.open()
        if not self.gripper.wait_until_executed():
            self.get_logger().error(f"Failed: {step}")
            return False
        return True

    def run(self):
        color = str(self.get_parameter("target_color").value).upper()
        target = TARGETS.get(color)
        if target is None:
            self.get_logger().error("target_color must be R, G, or B.")
            return
        if not target:
            self.get_logger().error(f"No calibration is available for {color}.")
            return

        self.get_logger().info(f"Starting {color} pick-and-place sequence.")

        # Pick.
        if not self.move("Home", target["home"]):
            return
        if not self.move("Pick step 1: move to pickup approach", target["pick_approach"]):
            return
        if not self.move("Pick step 2: lower shoulder", target["pick_grasp"]):
            return
        if not self.set_gripper("Pick step 3: close gripper halfway", close=True):
            return
        if not self.move("Pick step 4: lift", target["pick_lift"]):
            return

        # Drop.
        if not self.move("Drop step 1: move column to drop position", target["drop_approach"]):
            return
        if not self.move("Drop step 2: return arm joints", target["drop_release"]):
            return
        if not self.set_gripper("Drop step 3: open gripper", close=False):
            return
        if not self.move("Drop step 4: return column home", target["drop_home"]):
            return

        print("Pick-and-place sequence complete.")


def main():
    rclpy.init()
    node = PickAndPlace()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()

    try:
        node.create_rate(1.0).sleep()  # Allow MoveIt action clients to connect.
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Pick-and-place sequence interrupted.")
    except Exception as error:
        node.get_logger().error(f"Pick-and-place sequence failed: {error}")
    finally:
        executor.shutdown()
        thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
