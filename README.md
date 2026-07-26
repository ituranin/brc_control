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