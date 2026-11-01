import os
import subprocess
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _workspace_sim_dir():
    # <ws>/install/brc_control -> <ws>/brc-sim
    ws = os.path.dirname(os.path.dirname(get_package_prefix('brc_control')))
    return os.path.join(ws, 'brc-sim')


def _generate_track(sim_dir, track):
    """Regenerate brc-sim/models/track from a track JSON (a name in brc-sim/json or a path)."""
    json_dir = os.path.join(sim_dir, 'json')
    candidates = [track, os.path.join(json_dir, track), os.path.join(json_dir, track + '.json')]
    json_path = next((c for c in candidates if os.path.isfile(c)), None)
    if json_path is None:
        available = sorted(f[:-5] for f in os.listdir(json_dir) if f.endswith('.json'))
        raise RuntimeError(f"Track '{track}' not found. Available: {', '.join(available)}")
    subprocess.run(
        [sys.executable, os.path.join(sim_dir, 'generate_track_sdf.py'), json_path,
         '--out-dir', os.path.join(sim_dir, 'models', 'track')],
        cwd=sim_dir, check=True)


def _launch_setup(context):
    cfg = lambda name: LaunchConfiguration(name).perform(context)
    flag = lambda name: cfg(name).lower() in ('true', '1', 'yes')

    sim_dir = cfg('sim_dir')

    if cfg('track'):
        _generate_track(sim_dir, cfg('track'))

    gz_cmd = ['gz', 'sim', '-r', cfg('world')]
    if not flag('gui'):
        gz_cmd.insert(2, '-s')  # server only, no window

    actions = [
        # Started from brc-sim so the relative model:// paths of the world resolve
        ExecuteProcess(cmd=gz_cmd, cwd=sim_dir, output='screen'),
        Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            parameters=[{'config_file': os.path.join(sim_dir, 'config', 'bridge.yaml')}],
            output='screen'),
        Node(
            package='brc_control', executable='control',
            parameters=[cfg('params_file')], output='screen'),
    ]
    if flag('rviz'):
        actions.append(Node(
            package='rviz2', executable='rviz2', output='log',
            arguments=['-d', cfg('rviz_config')]))
    if flag('plot'):
        # PlotJuggler 4 has no direct ROS plugin; it connects through the Foxglove bridge
        actions.append(Node(package='foxglove_bridge', executable='foxglove_bridge', output='log'))
        actions.append(ExecuteProcess(cmd=['ros2', 'run', 'plotjuggler', 'plotjuggler'], output='log'))
    if flag('reconfigure'):
        actions.append(ExecuteProcess(
            cmd=['ros2', 'run', 'rqt_reconfigure', 'rqt_reconfigure'], output='log'))
    return actions


def generate_launch_description():
    share = get_package_share_directory('brc_control')
    return LaunchDescription([
        DeclareLaunchArgument('sim_dir', default_value=_workspace_sim_dir(),
                              description='Path of the brc-sim repository'),
        DeclareLaunchArgument('world', default_value='worlds/default.sdf',
                              description='World file, relative to sim_dir'),
        DeclareLaunchArgument('track', default_value='',
                              description='Regenerate the track first: name in brc-sim/json (e.g. fsg22_big) or a JSON path. Empty = keep the current track'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'configs', 'control.yaml')),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share, 'configs', 'dyn_pure_pursuit.rviz')),
        DeclareLaunchArgument('gui', default_value='true', description='Open the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('plot', default_value='true', description='Start the Foxglove bridge and PlotJuggler'),
        DeclareLaunchArgument('reconfigure', default_value='false', description='Start rqt_reconfigure'),
        OpaqueFunction(function=_launch_setup),
    ])
