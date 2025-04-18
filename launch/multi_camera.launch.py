from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cam1',
            name='realsense2_camera',
            output='screen',
            parameters=[{
                'serial_no': '241222073458',
                'enable_sync': True,
                'diagnostics_period': 1.0,
                'enable_color': True,
                'enable_depth': True,
                'color_width': 640,
                'color_height': 480,
                'color_fps': 30,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30
            }]
        )
    ,

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cam2',
            name='realsense2_camera',
            output='screen',
            parameters=[{
                'serial_no': '243522075041',
                'enable_sync': True,
                'diagnostics_period': 1.0,
                'enable_color': True,
                'enable_depth': True,
                'color_width': 640,
                'color_height': 480,
                'color_fps': 30,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30
            }]
        )
    ,

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='cam3',
            name='realsense2_camera',
            output='screen',
            parameters=[{
                'serial_no': '250122074372',
                'enable_sync': True,
                'diagnostics_period': 1.0,
                'enable_color': True,
                'enable_depth': True,
                'color_width': 640,
                'color_height': 480,
                'color_fps': 30,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30
            }]
        )
    ,
    ])
