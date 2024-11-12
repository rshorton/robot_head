from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time', 
            default_value='false',
            description='Set to true to use sim time'
        ),

        DeclareLaunchArgument(
            name='cmd_vel_topic', 
            default_value='cmd_vel',
            description='Topic to publish cmd_vel messages'
        ),        

        DeclareLaunchArgument(
            name='use_video_server', 
            default_value='false',
            description='Set to true to launch web video server'
        ),

        DeclareLaunchArgument(
            name='video_server_port', 
            default_value='8095',
            description='Port to use for web video server'
        ),

        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),

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
        ),

        Node(
            condition=IfCondition(LaunchConfiguration('use_video_server')),
            package='web_video_server',
            executable='web_video_server',
            parameters=[{'port': LaunchConfiguration('video_server_port')}]
        )
    ])