## Control System Case Study

In the previous section, we gained a preliminary understanding of the robot's control system and learned that machine learning has many interesting and promising research directions in the field of robotic control systems.
However, due to the complexity of control systems and the forward-looking nature of this research, they are not well suited as simple case studies.

At the same time, as a mature robotics framework, ROS already includes many mature and stable classical control components.
These control components, together with other mature functional modules, form larger-scale functional modules to accomplish more complex tasks.

Among these larger-scale functional modules, **Nav2** and **MoveIt2** are perhaps the two most commonly used.

As the names suggest, both of these functional modules are successors to their ROS1 versions.
Nav2 is the successor to the ROS Navigation Stack in ROS2, focusing on navigation-related functionality for mobile robots, such as localization and path planning, and is dedicated to safely moving a robot from one point to another.
MoveIt2 is the successor to ROS MoveIt in ROS2, dedicated to building an easy-to-use robotic manipulation platform. Robots with robotic arms essentially cannot do without it.

Both modules are mature, reliable, and easy to use. When developing robots using the ROS framework, developers will generally use them directly or make custom modifications on top of their existing functionality to avoid reinventing the wheel.

Therefore, in this section, we will use Nav2 as a case study to give readers a preliminary understanding of how to use a large-scale ROS2 functional module.

The content of this section largely references Nav2's [official documentation](https://navigation.ros.org/), especially the "Getting Started" chapter. Readers who are confident in their English can try reading the official documentation for more details.

There is no additional code example for this chapter.

### Installation

First, let us install the Nav2-related packages through Ubuntu's package manager.

```shell
sudo apt install ros-foxy-navigation2 ros-foxy-nav2-bringup
```

Here, `ros-foxy-navigation2` is Nav2's core package, and `ros-foxy-nav2-bringup` is a Nav2 launch example.
This example is quite flexible, and in many cases we can slightly modify it and use it in our own projects.

Next, let us install the `turtlebot3` series of packages.
The TurtleBot series is a very successful entry-level mobile robot series.
These packages provide components related to the TurtleBot3 robot, including functional components for using a virtual TurtleBot3 robot in a simulated environment.

```shell
sudo apt install "ros-foxy-turtlebot3*"
```

### Running

After installing the packages above, we can try using Nav2.

First, let us open a new terminal window and execute the following commands. These commands source the ROS2 framework, set which TurtleBot3 model to use, and configure where to find the models needed for the virtual world (Gazebo).

```shell
source /opt/ros/foxy/setup.bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/foxy/share/turtlebot3_gazebo/models
```

Now, everything is ready, and we can run a Nav2 demo program with the following command.

```shell
ros2 launch nav2_bringup tb3_simulation_launch.py
```

Here, the `ros2 launch` command is used to execute a launch file, which is a specification file that gathers many ROS2 components that need to be started and launches them according to plan.
A robot project often needs to start many different components to work together to accomplish tasks.
If each component requires opening a new window and executing a command, the entire robot startup process becomes very tedious.
Launch files and the `ros2 launch` command solve this problem.
We can think of the entire ROS2 project as a symphony orchestra, where each component represents an instrument.
The launch file is like the orchestra conductor, responsible for coordinating when each instrument should start playing.
In short, this is a very practical feature of ROS2.

For more details about the `ros2 launch` command and launch files, interested readers can consult the [official documentation](https://docs.ros.org/en/foxy/Tutorials/Launch/Creating-Launch-Files.html).

After successfully running the above command, we should see two newly opened GUI windows, corresponding to the `RViz` and `Gazebo` programs.
`RViz` is the visualization interface of the ROS2 framework, and we will use it later to control our virtual robot.
`Gazebo` is software for creating and running virtual worlds.
It is independent of the ROS2 framework, but the two work closely together.

In the `Gazebo` window (as shown in the figure below), we should be able to see a three-dimensional hexagonal virtual world.
This world also contains a virtual TurtleBot3 robot.
The robot emits many blue rays.
These rays represent the LiDAR reading rays of the robot.
The LiDAR readings are used by Nav2 to localize the robot in the environment.

![Gazebo Screenshot 1](../img/ch13/ros2-gazebo-1.JPG)

In the `RViz` window (as shown in the figure below), we should be able to see a two-dimensional map of the virtual world.
The white areas on the map are parts the robot can reach, while the black areas are detected obstacles or walls.
If you see a red `Global Status: Error` on the left side, your robot has not been correctly localized in RViz (i.e., the ROS2 framework).
Please select `2D Pose Estimate` from the toolbar and update the robot's pose at the position where the robot should be on the RViz map (using the robot's position in Gazebo as the reference).

![RViz Screenshot 1](../img/ch13/ros2-rviz-1.JPG)

After updating the robot's pose, RViz should look similar to the figure below.

![RViz Screenshot 2](../img/ch13/ros2-rviz-2.JPG)

With this, our robot is ready to move in the virtual world.

Please select the `Navigation2 Goal` button from the RViz toolbar and choose the position and pose you want the TurtleBot3 robot to reach on the map.
Once selected, you will see the robot start moving toward the target position and eventually arrive at the destination.

RViz also provides buttons for many other Nav2 features, which you can learn more about through Nav2 and ROS2's official documentation.

Congratulations, you now have a preliminary understanding of how to use large-scale functional modules within the ROS2 framework!

#### Chapter Appendix: Using Nav2 in WSL

Some readers may be running ROS2 through WSL (Windows Subsystem for Linux) on Windows.
If this is the case, the graphical interface programs in this chapter, such as RViz and Gazebo, may cause issues.
This is because WSL does not support running graphical interface programs by default.

Fortunately, we can change settings to enable running graphical interface programs in WSL.
[This guide](https://github.com/rhaschke/lecture/wiki/WSL-install) describes how the author runs ROS2 and graphical interfaces in WSL. The second point is particularly noteworthy.
And [this guide](https://github.com/cascadium/wsl-windows-toolbar-launcher#firewall-rules) provides more detailed instructions on how to run graphical interface programs in WSL in general.

These two guides should provide readers with enough information to resolve the issues mentioned above related to RViz and Gazebo. The only drawback is that both guides are in English, which requires a certain level of English proficiency from readers.
