#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class LegController(Node):
    def __init__(self):
        super().__init__('leg1_joint_commander')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/leg1_controller/follow_joint_trajectory')

    def send_goal(self):
        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_11', 'joint_12']

        point = JointTrajectoryPoint()
        point.positions = [0.3, -0.3]  # 要移动的目标角度（单位：弧度）
        point.time_from_start.sec = 2  # 2秒内完成动作

        goal_msg.trajectory.points.append(point)

        self.get_logger().info("Sending goal...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected 😥')
            return

        self.get_logger().info('Goal accepted 😊')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action completed with status: {result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LegController()
    node.send_goal()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

