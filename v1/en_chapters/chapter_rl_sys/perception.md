## Perception System

The perception system includes not only visual perception but also tactile sensing, auditory sensing, and more. In unknown environments, for a robot to achieve autonomous movement and navigation, it must know where it is (through camera relocalization :cite:`ding2019camnet`), what the surroundings look like (through 3D object detection :cite:`yi2020segvoxelnet` or semantic segmentation), and predict the camera trajectory in space :cite:`9813561`. All of these rely on the perception system :cite:`xu2019depth`.
When it comes to perception systems, Simultaneous Localization and Mapping (SLAM) is an indispensable topic. The SLAM process generally includes landmark extraction, data association, state estimation, state update, and landmark update. Visual Odometry is an important component of SLAM, estimating the relative motion (ego-motion) of the robot between two time steps. The ORB-SLAM series is a representative work in visual SLAM. :numref:`orbslam3` shows the main system components of the latest ORB-SLAM3. VINS-Mono, an open-source monocular visual-inertial SLAM system from the Hong Kong University of Science and Technology, is also worth noting. Multi-sensor fusion, optimizing data association and loop closure detection, integration with front-end heterogeneous processors, and improving robustness and relocalization accuracy are all future development directions for SLAM technology.

Recently, with the rise of machine learning, learning-based SLAM frameworks have also been proposed. TartanVO is the first learning-based Visual Odometry (VO) model that can generalize to multiple datasets and real-world scenarios, outperforming traditional geometry-based methods.
UnDeepVO is an unsupervised deep learning approach that can estimate the 6-DoF pose of a monocular camera and its view depth using deep neural networks. DROID-SLAM is a deep visual SLAM system for monocular, stereo, and RGB-D cameras. Through iterative updates of camera poses and pixel depths via a Bundle Adjustment layer, it achieves strong robustness with significantly reduced failures. Although trained on monocular video, it can leverage stereo or RGB-D video at test time to improve performance. The combination of Bundle Adjustment (BA) and machine learning has been extensively studied. CMU proposed a modular system through active neural SLAM to help intelligent robots explore unknown environments efficiently.

### Object Detection and Semantic Segmentation

The perception system includes not only visual perception but also tactile sensing, auditory sensing, and more. In unknown environments, for a robot to achieve autonomous movement and navigation, it must know where it is (through camera relocalization :cite:`ding2019camnet`), what the surroundings look like (through 3D object detection :cite:`yi2020segvoxelnet` or semantic segmentation), and predict the camera trajectory in space :cite:`9813561`. All of these rely on the perception system :cite:`xu2019depth`.

Image semantic segmentation, as a commonly used and classical perception technique, has gradually matured in the traditional 2D domain after years of continuous iteration, with limited room for improvement. At the same time, traditional 2D semantic segmentation has certain limitations: it is difficult to directly obtain the spatial position of objects and their layout in the overall space from 2D images. To obtain the position information of the entire space, more three-dimensional information is needed. To enable robots to obtain the 3D coordinates, semantics, and boundary information of objects in space from purely 2D images, cross-view semantic segmentation :cite:`9123682` has attracted the attention of many researchers.

### Simultaneous Localization and Mapping (SLAM)

When a robot is placed in an unknown environment, how can it understand its own position and the surrounding environment? This is achieved through Simultaneous Localization and Mapping (SLAM) systems.

:numref:`orbslam3` shows the main system components of the latest ORB-SLAM3.
The SLAM process generally includes landmark extraction, data association, state estimation, state update, and landmark update. During the robot's movement, the SLAM system locates its own position and orientation by repeatedly observing map features (such as wall corners, pillars, etc.), and then incrementally constructs the map based on its position, thereby achieving the goal of simultaneous localization and map construction.

DROID-SLAM is a deep visual SLAM system for monocular, stereo, and RGB-D cameras. Through iterative updates of camera poses and pixel depths via a Bundle Adjustment layer, it achieves strong robustness with significantly reduced failures. Although trained on monocular video, it can leverage stereo or RGB-D video at test time to improve performance.
Bundle Adjustment (BA) describes the sum of errors between pixel coordinates and reprojected coordinates, where reprojected coordinates are typically computed using 3D coordinate points and camera parameters. BA is computationally intensive and time-consuming. The University of Edinburgh proposed accelerating BA computation through a distributed multi-GPU system :cite:`MegBA`. With the development of machine learning, the combination of BA and machine learning has been widely studied.

Visual Odometry is an important component of SLAM, estimating the relative motion of the robot between two time steps.
Recently, with the rise of machine learning, learning-based VO frameworks have also been proposed.
TartanVO is the first learning-based Visual Odometry (VO) model that can generalize to multiple datasets and real-world scenarios, outperforming traditional geometry-based methods.

![Main System Components of ORB-SLAM3 :cite:`campos2021orb`](../img/ch13/orbslam3.png)

:width:`800px`

:label:`orbslam3`

