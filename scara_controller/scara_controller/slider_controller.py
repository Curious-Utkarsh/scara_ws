#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SliderControl(Node):
    def __init__(self):
        super().__init__("slider_control")
        self.arm_pub_ = self.create_publisher(JointTrajectory, "arm_controller/joint_trajectory", 10)
        self.gripper_pub_ = self.create_publisher(JointTrajectory, "gripper_controller/joint_trajectory", 10)
        self.sub_ = self.create_subscription(JointState, "joint_commands", self.sliderCallback, 10)
        self.get_logger().info("Slider Control Node started")

    def sliderCallback(self, msg):
        arm_msg = JointTrajectory()
        gripper_msg = JointTrajectory()

        arm_msg.joint_names = [
            "column_joint",
            "shoulder_joint",
            "forearm_joint",
            "wrist_joint",
        ]
        gripper_msg.joint_names = ["left_finger_joint"]

        arm_goal = JointTrajectoryPoint()
        gripper_goal = JointTrajectoryPoint()

        arm_goal.positions = list(msg.position[:4])
        gripper_goal.positions = [msg.position[4]]

        # Required: controller rejects points with zero time_from_start
        arm_goal.time_from_start = Duration(sec=1, nanosec=0)
        gripper_goal.time_from_start = Duration(sec=1, nanosec=0)

        arm_msg.points.append(arm_goal)
        gripper_msg.points.append(gripper_goal)

        self.arm_pub_.publish(arm_msg)
        self.gripper_pub_.publish(gripper_msg)


def main():
    rclpy.init()
    simple_publisher = SliderControl()
    rclpy.spin(simple_publisher)
    simple_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()