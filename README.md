brc_control
===========

This package is an autonomous controller for brc-sim. It uses a BEV road segmentation to get the target heading from the centerline of the road. Speed is controlled via a software radar that works on the BEV and gets the target speed based on distance to road border.

## Compilation and start

Use the following commands in your ros workspace folder. Make sure that this repo is placed in the src folder of the workspace.

```
colcon build --packages-select brc_control
source install/setup.bash
```

```
ros2 run brc_control control
```
## Launch everything

Build once with `--symlink-install`, so edits to the launch file, the parameter file and the Python code are picked up without rebuilding (new files still need a rebuild):

```
colcon build --symlink-install --packages-select brc_control
source install/setup.bash
ros2 launch brc_control brc.launch.py
```

This starts Gazebo, the ROS bridge, the control node, rviz, the Foxglove bridge and PlotJuggler (select "Foxglove ROS2 Bridge" under Streaming). Gazebo's stop/reset/run buttons keep working while the other nodes stay up.

| Argument | Default | Meaning |
|---|---|---|
| `track` | empty | Regenerate `brc-sim/models/track` from a JSON first: a name in `brc-sim/json` (e.g. `fsg22_big`) or a path. Empty keeps the current track. Needs a relaunch, a Gazebo reset is not enough |
| `gui` / `rviz` / `plot` | `true` | Gazebo window / rviz / Foxglove bridge + PlotJuggler |
| `reconfigure` | `false` | Also open `rqt_reconfigure` |
| `params_file` | `configs/control.yaml` | Controller parameters |
| `rviz_config` | `configs/dyn_pure_pursuit.rviz` | rviz layout |
| `sim_dir`, `world` | `<ws>/brc-sim`, `worlds/default.sdf` | Where the sim lives |

## Tuning without restarting

All numeric controller settings are ROS parameters of the node `brc_control` (see `configs/control.yaml`). Change them live, for example in `rqt_reconfigure` (`ros2 launch brc_control brc.launch.py reconfigure:=true`) or with `ros2 param set /brc_control look_ahead_descent_limit 0.8`. `look_ahead_lut` is a flat list of `[curvature, look_ahead, ...]` pairs. Live changes are not saved: write them to `configs/control.yaml` (or run `ros2 param dump /brc_control`) and restart the node. `lateral_controller` is only read at startup.

## Live tuning with rqt_reconfigure

Open it together with everything else:

```
ros2 launch brc_control brc.launch.py reconfigure:=true
```

or, with the sim and the control node already running, in a separate terminal:

```
source /opt/ros/lyrical/setup.bash
source ~/src/brc_ws/install/setup.bash
ros2 run rqt_reconfigure rqt_reconfigure
```

1. Click **brc_control** in the node list on the left. If it is missing, press the refresh button; the node has to be running.
2. Change a value with the spinbox, slider or checkbox. It takes effect on the next control step, so keep the plots open to watch the car react.

Notes:

- `look_ahead_lut` (flat list of `[curvature, look_ahead, ...]` pairs) can be awkward to edit in the GUI. Set it from the command line instead:
  ```
  ros2 param set /brc_control look_ahead_lut "[0.1, 4.0, 0.01, 8.0, 0.001, 12.0]"
  ```
  A list with fewer than two pairs is rejected.
- Nothing is saved. Run `ros2 param dump /brc_control` and copy the values into `configs/control.yaml` so they survive a restart.
- `lateral_controller` is only read at startup, so changing it live has no effect.
- A value in `configs/control.yaml` overrides the default in the controller class at startup. To change a default in the code, remove that parameter from the YAML or edit it there, and restart the node (a running node never reloads its code).
