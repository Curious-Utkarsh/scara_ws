#!/usr/bin/env python3
"""Two-hand teleop node for the SCARA robot.

Replaces the joint_state_publisher_gui sliders used by slider_controller.py
with a webcam + MediaPipe Hands pipeline, using BOTH hands for full 4-DOF
control plus the gripper:

    LEFT hand
        - palm-center X (side to side) -> column_joint   (base rotation)
        - palm-center Y (up / down)     -> shoulder_joint (Z-lift)

    RIGHT hand
        - palm-center X (side to side) -> forearm_joint   (arm sweep)
        - palm-center Y (up / down)     -> wrist_joint     (end-effector rotation)
        - thumb-tip <-> index-tip pinch -> left_finger_joint (gripper)

Each hand is only used while it's visible in frame; if a hand leaves the
frame, its joints hold their last commanded value instead of snapping back
to a default pose.

The debug window shows a plain, uncluttered HUD: a hand skeleton per hand
(steel blue = left, amber = right), the active tracking zone, and a row of
simple gauges showing every joint's current position within its limits.

Requires (pip): opencv-python, mediapipe
"""
import math

import cv2
import mediapipe as mp
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Joint limits, taken from scara_description/urdf/arm.xacro
COLUMN_LIMITS = (-2.0, 2.0)          # column_joint (revolute)      - left hand X
FOREARM_LIMITS = (-2.0, 2.0)         # forearm_joint (revolute)     - right hand X
GRIPPER_LIMITS = (-0.06, 0.0)        # left_finger_joint: -0.02 = closed, 0.0 = open

# shoulder_joint / wrist_joint limits weren't in the original slider config,
# these are placeholders - double check them against arm.xacro and adjust
# the shoulder_limit_min/max and wrist_limit_min/max ROS parameters below.
SHOULDER_LIMITS = (-0.1, 0.05)      # shoulder_joint (prismatic Z-lift) - left hand Y
WRIST_LIMITS = (-2.0, 2.0)           # wrist_joint (revolute)             - right hand Y

# MediaPipe hand-landmark indices used below
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
THUMB_TIP = 4
INDEX_TIP = 8

PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# ---- HUD color palette (BGR, since everything is drawn with cv2) ----------
LEFT_HAND_COLOR = (170, 120, 60)    # muted steel blue  -> column / shoulder
RIGHT_HAND_COLOR = (60, 140, 190)   # muted amber       -> forearm / wrist / gripper
UI_COLOR = (210, 210, 210)          # light gray for chrome/text
DIM_COLOR = (100, 100, 100)         # dim gray for "not detected" state
PANEL_COLOR = (25, 25, 25)          # neutral dark translucent panel
BAR_BG_COLOR = (55, 55, 55)         # gauge background


def ema(previous, new, alpha):
    """Exponential moving average filter (alpha = weight on the new sample)."""
    return new if previous is None else (alpha * new + (1.0 - alpha) * previous)


def normalized_to_range(value, in_min, in_max, out_min, out_max, invert=False):
    """Linearly map `value` from [in_min, in_max] into [out_min, out_max], clamped."""
    value = min(max(value, in_min), in_max)
    ratio = (value - in_min) / (in_max - in_min) if in_max != in_min else 0.0
    if invert:
        ratio = 1.0 - ratio
    return out_min + ratio * (out_max - out_min)


class HandGestureControl(Node):
    def __init__(self):
        super().__init__("hand_gesture_control")

        # ---- parameters (tune without touching code) ---------------------
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("active_margin", 0.15)
        self.declare_parameter("invert_column", False)
        self.declare_parameter("invert_shoulder", True)
        self.declare_parameter("invert_forearm", False)
        self.declare_parameter("invert_wrist", False)
        self.declare_parameter("invert_gripper", False)
        self.declare_parameter("pinch_ratio_min", 0.35)
        self.declare_parameter("pinch_ratio_max", 1.3)
        self.declare_parameter("shoulder_limit_min", SHOULDER_LIMITS[0])
        self.declare_parameter("shoulder_limit_max", SHOULDER_LIMITS[1])
        self.declare_parameter("wrist_limit_min", WRIST_LIMITS[0])
        self.declare_parameter("wrist_limit_max", WRIST_LIMITS[1])
        self.declare_parameter("swap_handedness", False)
        self.declare_parameter("show_debug_window", True)
        self.declare_parameter("min_detection_confidence", 0.6)
        self.declare_parameter("min_tracking_confidence", 0.6)
        self.declare_parameter("trajectory_time_from_start_sec", 0.15)

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.camera_index = g("camera_index")
        self.publish_rate_hz = g("publish_rate_hz")
        self.alpha = g("smoothing_alpha")
        self.active_margin = g("active_margin")
        self.invert_column = g("invert_column")
        self.invert_shoulder = g("invert_shoulder")
        self.invert_forearm = g("invert_forearm")
        self.invert_wrist = g("invert_wrist")
        self.invert_gripper = g("invert_gripper")
        self.pinch_ratio_min = g("pinch_ratio_min")
        self.pinch_ratio_max = g("pinch_ratio_max")
        self.shoulder_limits = (g("shoulder_limit_min"), g("shoulder_limit_max"))
        self.wrist_limits = (g("wrist_limit_min"), g("wrist_limit_max"))
        self.swap_handedness = g("swap_handedness")
        self.show_debug_window = g("show_debug_window")
        self.time_from_start_sec = g("trajectory_time_from_start_sec")

        # ---- publishers (same topics slider_controller.py uses) ----------
        self.arm_pub_ = self.create_publisher(JointTrajectory, "arm_controller/joint_trajectory", 10)
        self.gripper_pub_ = self.create_publisher(JointTrajectory, "gripper_controller/joint_trajectory", 10)

        # ---- camera + MediaPipe --------------------------------------------
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera index {self.camera_index}")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=g("min_detection_confidence"),
            min_tracking_confidence=g("min_tracking_confidence"),
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # ---- smoothed per-hand feature state ---------------------------------
        self._left_palm_x = None
        self._left_palm_y = None
        self._right_palm_x = None
        self._right_palm_y = None
        self._pinch_ratio = None
        self._left_active = False
        self._right_active = False

        # Last commanded targets. Holding these steady (instead of resetting
        # to zero) whenever a hand isn't visible avoids the arm jerking back
        # to a default pose every time your hand briefly leaves the frame.
        self._column_pos = 0.0
        self._shoulder_pos = min(max(0.0, self.shoulder_limits[0]), self.shoulder_limits[1])
        self._forearm_pos = 0.0
        self._wrist_pos = 0.0
        self._gripper_pos = GRIPPER_LIMITS[1]  # start open

        period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(period, self.update)
        self.get_logger().info("Hand Gesture Control Node started (two-hand mode)")

    # -------------------------------------------------------------------------
    def update(self):
        if not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read frame from camera", throttle_duration_sec=2.0)
            return

        frame = cv2.flip(frame, 1)  # mirror, so hand motion matches what you see on screen
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self._left_active = False
        self._right_active = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                if self.swap_handedness:
                    label = "Right" if label == "Left" else "Left"

                if label == "Left":
                    self._left_active = True
                    self._process_left(landmarks.landmark)
                    hand_color = LEFT_HAND_COLOR
                else:
                    self._right_active = True
                    self._process_right(landmarks.landmark)
                    hand_color = RIGHT_HAND_COLOR

                if self.show_debug_window:
                    spec = self.mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=3)
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS, spec, spec)

        self._publish_targets()

        if self.show_debug_window:
            self._draw_debug_overlay(frame)
            scale = 1.8
            display = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.imshow("SCARA hand control", display)
            # cv2.imshow("SCARA hand control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed, shutting down hand control node")
                rclpy.shutdown()

    # -------------------------------------------------------------------------
    def _process_left(self, lm):
        """Left palm position -> column_joint (X) and shoulder_joint (Y)."""
        palm_x = sum(lm[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
        palm_y = sum(lm[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)

        self._left_palm_x = ema(self._left_palm_x, palm_x, self.alpha)
        self._left_palm_y = ema(self._left_palm_y, palm_y, self.alpha)

        m = self.active_margin
        self._column_pos = normalized_to_range(
            self._left_palm_x, m, 1.0 - m, COLUMN_LIMITS[0], COLUMN_LIMITS[1], invert=self.invert_column
        )
        self._shoulder_pos = normalized_to_range(
            self._left_palm_y, m, 1.0 - m, self.shoulder_limits[0], self.shoulder_limits[1],
            invert=self.invert_shoulder,
        )

    # -------------------------------------------------------------------------
    def _process_right(self, lm):
        """Right palm position -> forearm_joint (X) and wrist_joint (Y);
        thumb/index pinch -> gripper."""
        palm_x = sum(lm[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
        palm_y = sum(lm[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)

        # Normalize pinch distance by hand size so it isn't affected by
        # how close/far your hand is from the camera.
        hand_size = math.hypot(lm[MIDDLE_MCP].x - lm[WRIST].x, lm[MIDDLE_MCP].y - lm[WRIST].y)
        pinch_dist = math.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x, lm[THUMB_TIP].y - lm[INDEX_TIP].y)
        pinch_ratio = pinch_dist / hand_size if hand_size > 1e-6 else self.pinch_ratio_max

        self._right_palm_x = ema(self._right_palm_x, palm_x, self.alpha)
        self._right_palm_y = ema(self._right_palm_y, palm_y, self.alpha)
        self._pinch_ratio = ema(self._pinch_ratio, pinch_ratio, self.alpha)

        m = self.active_margin
        self._forearm_pos = normalized_to_range(
            self._right_palm_x, m, 1.0 - m, FOREARM_LIMITS[0], FOREARM_LIMITS[1], invert=self.invert_forearm
        )
        self._wrist_pos = normalized_to_range(
            self._right_palm_y, m, 1.0 - m, self.wrist_limits[0], self.wrist_limits[1], invert=self.invert_wrist
        )
        # Pinching fingers together (small ratio) -> closed; spreading them
        # apart (large ratio) -> open. That's the natural gesture direction.
        self._gripper_pos = normalized_to_range(
            self._pinch_ratio, self.pinch_ratio_min, self.pinch_ratio_max,
            GRIPPER_LIMITS[0], GRIPPER_LIMITS[1], invert=self.invert_gripper,
        )

    # -------------------------------------------------------------------------
    def _publish_targets(self):
        duration = Duration(
            sec=int(self.time_from_start_sec),
            nanosec=int((self.time_from_start_sec % 1.0) * 1e9),
        )

        arm_msg = JointTrajectory()
        arm_msg.joint_names = ["column_joint", "shoulder_joint", "forearm_joint", "wrist_joint"]
        arm_goal = JointTrajectoryPoint()
        arm_goal.positions = [
            self._column_pos,
            self._shoulder_pos,
            self._forearm_pos,
            self._wrist_pos,
        ]
        arm_goal.time_from_start = duration
        arm_msg.points.append(arm_goal)

        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = ["left_finger_joint"]
        gripper_goal = JointTrajectoryPoint()
        gripper_goal.positions = [self._gripper_pos]
        gripper_goal.time_from_start = duration
        gripper_msg.points.append(gripper_goal)

        self.arm_pub_.publish(arm_msg)
        self.gripper_pub_.publish(gripper_msg)

    # ==================== HUD / debug-view rendering ==========================
    def _draw_gauge(self, frame, x, y, bar_w, bar_h, value, vmin, vmax, label, color, active):
        ratio = 0.0 if vmax == vmin else min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
        fill_h = int(ratio * bar_h)
        draw_color = color if active else DIM_COLOR

        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), BAR_BG_COLOR, -1)
        cv2.rectangle(frame, (x, y + bar_h - fill_h), (x + bar_w, y + bar_h), draw_color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), UI_COLOR, 1)
        cv2.putText(frame, label, (x - 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{value:+.2f}", (x - 4, y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_COLOR, 1,
                    cv2.LINE_AA)

    def _draw_debug_overlay(self, frame):
        h, w = frame.shape[:2]

        # plain translucent bars, top (title/status) and bottom (gauges)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), PANEL_COLOR, -1)
        cv2.rectangle(overlay, (0, h - 112), (w, h), PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

        # active-zone box, where palm tracking maps to the full joint range
        m = self.active_margin
        cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)), UI_COLOR, 1)

        # title + hand status, plain text
        cv2.putText(frame, "SCARA Hand Control", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLOR, 1, cv2.LINE_AA)
        left_status = "Left hand: active" if self._left_active else "Left hand: not detected"
        right_status = "Right hand: active" if self._right_active else "Right hand: not detected"
        left_color = LEFT_HAND_COLOR if self._left_active else DIM_COLOR
        right_color = RIGHT_HAND_COLOR if self._right_active else DIM_COLOR
        cv2.putText(frame, left_status, (w - 250, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1, cv2.LINE_AA)
        cv2.putText(frame, right_status, (w - 250, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1, cv2.LINE_AA)

        # joint gauges along the bottom
        gauges = [
            ("column", self._column_pos, COLUMN_LIMITS, LEFT_HAND_COLOR, self._left_active),
            ("shoulder", self._shoulder_pos, self.shoulder_limits, LEFT_HAND_COLOR, self._left_active),
            ("forearm", self._forearm_pos, FOREARM_LIMITS, RIGHT_HAND_COLOR, self._right_active),
            ("wrist", self._wrist_pos, self.wrist_limits, RIGHT_HAND_COLOR, self._right_active),
            ("gripper", self._gripper_pos, GRIPPER_LIMITS, RIGHT_HAND_COLOR, self._right_active),
        ]
        n = len(gauges)
        bar_w, bar_h = 40, 68
        gap = (w - n * bar_w) / (n + 1)
        for i, (label, value, limits, color, active) in enumerate(gauges):
            x = int(gap + i * (bar_w + gap))
            self._draw_gauge(frame, x, h - 96, bar_w, bar_h, value, limits[0], limits[1], label, color, active)

    # -------------------------------------------------------------------------
    def destroy_node(self):
        self.cap.release()
        self.hands.close()
        if self.show_debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = HandGestureControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()