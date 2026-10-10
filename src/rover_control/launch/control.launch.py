"""
control.launch.py
Launches odom_node and lqr_node together.

Usage:
    ros2 launch rover_control control.launch.py
    ros2 launch rover_control control.launch.py config:=/path/to/rover_params.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Allow config path override from CLI
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=PathJoinSubstitution([
            FindPackageShare('rover_control'), 'config', 'rover_params.yaml'
        ]),
        description='Path to rover_params.yaml'
    )

    config = LaunchConfiguration('config')

    odom_node = Node(
        package='rover_control',
        executable='odom_node',
        name='odom_node',
        arguments=[config],
        output='screen',
    )

    lqr_node = Node(
        package='rover_control',
        executable='lqr_node',
        name='lqr_node',
        arguments=[config],
        output='screen',
    )

    return LaunchDescription([
        config_arg,
        odom_node,
        lqr_node,
    ])
