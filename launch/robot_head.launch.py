from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_head',
            executable='vision',
        ),
        Node(
            package='robot_head',
            executable='tracker',
        ),
        Node(
            package='robot_head',
            executable='viewer',
        ),
        Node(
            package='robot_head',
            executable='face',
        )
    ])