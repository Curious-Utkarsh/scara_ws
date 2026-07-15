#!/usr/bin/env python3
"""Execute calibrated SCARA pick-and-drop routines requested by voice."""

from queue import Empty, Queue

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String

from pymoveit2 import GripperInterface, MoveIt2
from pymoveit2.robots import scara


# Joint order: column_joint, shoulder_joint, forearm_joint, wrist_joint.
# Keep these values in sync with pymoveit2/examples/pick_and_place.py.
TARGETS = {
    "R": {
        "home": [0.0, 0.0, 0.0, 0.0],
        "pick_approach": [0.0, -0.15, 0.0, 0.0],
        "pick_grasp": [0.0, -0.15, 0.0, 0.0],
        "pick_lift": [0.0, 0.0, 0.0, 0.0],
        "drop_approach": [-1.611, 0.0, -2.00, 0.0],
        "drop_release": [-1.611, -0.05, -2.00, 0.0],
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


class PickAndDrop(Node):
    """Listen for G, B, or R and perform one routine at a time."""

    def __init__(self):
        super().__init__("pick_and_drop")
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
            open_gripper_joint_positions=(
                scara.OPEN_GRIPPER_JOINT_POSITIONS
            ),
            closed_gripper_joint_positions=(
                scara.CLOSED_GRIPPER_JOINT_POSITIONS
            ),
            gripper_group_name=scara.MOVE_GROUP_GRIPPER,
            callback_group=callback_group,
        )
        self.subscription = self.create_subscription(
            String, "/scara/pick_command", self.command_callback, 10
        )
        self.commands = Queue()
        self.get_logger().info(
            "Listening on /scara/pick_command for G, B, or R."
        )

    def command_callback(self, msg):
        color = msg.data.strip().upper()
        if color not in TARGETS:
            self.get_logger().warning(
                "Ignoring command '%s'; expected G, B, or R." % msg.data
            )
            return
        self.commands.put(color)
        self.get_logger().info("Queued %s pick-and-drop command." % color)

    def process_next_command(self):
        """Run one queued command; leave the node ready for the next one."""
        try:
            color = self.commands.get_nowait()
        except Empty:
            return

        try:
            # A command starts only after the previous one has ended.
            self.arm.force_reset_executing_state()
            self.gripper.force_reset_executing_state()
            self.run_sequence(color)
        except Exception as error:
            # Keep listening if one MoveIt request fails.
            self.get_logger().error("Sequence for %s failed: %s" % (color, error))
        finally:
            self.commands.task_done()

    def move(self, step, joints):
        self.get_logger().info(step)
        self.arm.move_to_configuration(joints)
        if not self.arm.wait_until_executed():
            self.get_logger().error("Failed: %s" % step)
            return False
        return True

    def set_gripper(self, step, close):
        self.get_logger().info(step)
        self.gripper.close() if close else self.gripper.open()
        if not self.gripper.wait_until_executed():
            self.get_logger().error("Failed: %s" % step)
            return False
        return True

    def run_sequence(self, color):
        target = TARGETS[color]
        self.get_logger().info("Starting %s pick-and-drop sequence." % color)
        steps = (
            (self.move, "Home", target["home"]),
            (self.move, "Pick approach", target["pick_approach"]),
            (self.move, "Lower to grasp", target["pick_grasp"]),
            (self.set_gripper, "Close gripper", True),
            (self.move, "Lift object", target["pick_lift"]),
            (self.move, "Move to drop position", target["drop_approach"]),
            (self.move, "Lower at drop position", target["drop_release"]),
            (self.set_gripper, "Open gripper", False),
            (self.move, "Return home", target["drop_home"]),
        )
        for operation, step, value in steps:
            if not operation(step, value):
                return
        self.get_logger().info("%s pick-and-drop sequence complete." % color)


def main(args=None):
    rclpy.init(args=args)
    node = PickAndDrop()
    try:
        while rclpy.ok():
            # This is the only executor. MoveIt also uses spin_once() while a
            # trajectory runs, so topic commands received during motion queue up.
            rclpy.spin_once(node, timeout_sec=0.1)
            node.process_next_command()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
