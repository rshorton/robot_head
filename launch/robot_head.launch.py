from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time', 
            default_value='false',
            description='Enable use_sime_time to true'
        ),

        DeclareLaunchArgument(
            name='cmd_vel_topic', 
            default_value='cmd_vel',
            description='Topic to publish cmd_vel messages'
        ),        

        SetParameter(name='use_sim_time', value=LaunchConfiguration("use_sim_time")),

        Node(
            package='robot_head',
            executable='vision',
        ),

        Node(
            package='robot_head',
            executable='tracker',
            remappings=[('cmd_vel', LaunchConfiguration('cmd_vel_topic'))]
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