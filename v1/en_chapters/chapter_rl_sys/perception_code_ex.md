## Perception System Case Study

In the previous chapter on [Introduction to Robot Operating System (ROS)](./ros_code_ex.md), we learned how to create a ROS2 project and how to use nodes, services, actions, and other components within the ROS2 framework. Then, in the previous section, we gained a preliminary understanding of the robot's perception system.
In this section, we will demonstrate how to combine ROS2 and the deep learning framework PyTorch to implement a basic function of our envisioned perception system through a simple case study.

### Case Background

Suppose we want to design a fully automatic pineapple-picking robot for an orchard.
This robot may need an intelligent mobile base for moving around the orchard, several sensors (including an RGB camera) for detecting pineapples, and a robotic arm for the picking action.
Among the long list of functions this robot needs to perform, its perception system must be able to detect whether there is a pineapple in the center of the camera sensor's view.
Only after detecting a pineapple will the picking process begin.

This function of detecting whether a pineapple is present in the center of the image is a basic yet essential function of our robot's perception system.
Fortunately, with the development of modern convolutional neural networks, we can leverage existing deep learning frameworks, such as PyTorch, to quickly implement this function.
Moreover, a simple AlexNet pre-trained on ImageNet is already sufficient.

In previous case studies, we used only libraries within the ROS2 framework. In this example, we will begin to understand how to use Python libraries outside the ROS2 framework within ROS2.

Similar to previous case studies, the code used in this section can be found in the `src/object_detector` folder of the book's related [ROS2 case study code repository](https://github.com/openmlsys/openmlsys-ros2).

### Project Setup

Let us continue using the ROS2 project framework we have already set up.
We only need to add a new ROS2 Python package to implement the desired functionality.
Therefore, let us go back to the `src` directory and create this Python package.

```shell
cd openmlsys-ros2/src
ros2 pkg create --build-type ament_python --node-name object_detector_node object_detector --dependencies rclpy std_msgs sensor_msgs cv_bridge opencv-python torch torchvision torchaudio
```

After creating the Python package, do not forget to update the `version`, `maintainer`, `maintainer_email`, `description`, and `license` fields in `package.xml` and `setup.py`.

Next, we need to install the `image_publisher` package within the ROS2 framework.
This package helps us simulate a single image as an image stream similar to a camera video feed.
When developing a real robot, we might be able to test our program on the actual machine, but for this case study, we can only use this `image_publisher` package and a few selected images to test our program.
In practice, even when developing functionality for a real robot, it is best to perform unit testing with images before actual testing.

We can simply install this `image_publisher` package through Ubuntu's `apt`, as it is a commonly used ROS2 framework package that has been packaged for installation via Ubuntu's package manager.

```shell
sudo apt install ros-foxy-image-publisher
```

For more information and usage instructions about the `image_publisher` package, you can check [its documentation](http://wiki.ros.org/image_publisher). This documentation is for the earlier ROS1 version, but since this package has not changed since then, all the features described in the documentation are the same as in the ROS2 version we are using.

Next, let us install `opencv-python`, `torch`, `torchvision`, and `torchaudio` in the ROS2 project's Python virtual environment.
For example, users using `pipenv` might run the command `pipenv install opencv-python torch torchvision torchaudio`.

Finally, let us save the following two images of a pineapple and an apple in `openmlsys-ros2/data`.
We will use these two images to verify that our program can detect pineapples and will not mistake a pineapple for an apple.

![Pineapple Image](../img/ch13/ros-pineapple.jpg)

:width:`256px`

:label:`ros2-pineapple`

![Apple Image](../img/ch13/ros-apple.jpg)

:width:`256px`

:label:`ros2-apple`

### Adding Code

The command we used to create the Python package should have already created the file `src/object_detector/object_detector/object_detector_node.py`. Now let us replace the existing content of this file with the following content.

```Python
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

import torch
import torchvision.models as models
from torchvision import transforms


class ObjectDetectorNode(Node):

    PINEAPPLE_CLASS_ID = 953

    def __init__(self):
        super().__init__('object_detector_node')
        self.detection_publisher = self.create_publisher(Bool, 'object_detected', 10)
        self.camera_subscriber = self.create_subscription(
            Image, 'camera_topic', self.camera_callback, 10,
        )
        self.alex_net = models.alexnet(pretrained=True)
        self.alex_net.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cv_bridge = CvBridge()
        self.declare_parameter('detection_class_id', self.PINEAPPLE_CLASS_ID)
        self.get_logger().info(f'Detector node is ready.')

    def camera_callback(self, msg: Image):
        self.get_logger().info(f'Received an image, ready to detect!')
        detection_class_id = self.get_parameter('detection_class_id').get_parameter_value().integer_value
        img = self.cv_bridge.imgmsg_to_cv2(msg)
        input_batch = self.preprocess(img).unsqueeze(0)
        img_output = self.alex_net(input_batch)[0]
        detection = Bool()
        detection.data = torch.argmax(img_output).item() == detection_class_id
        self.detection_publisher.publish(detection)
        self.get_logger().info(f'Detected: "{detection.data}", target class id: {detection_class_id}')


def main(args=None):
    rclpy.init(args=args)
    object_detector_node = ObjectDetectorNode()
    rclpy.spin(object_detector_node)
    object_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Observant readers may have already noticed that this code is very similar to our previous code for creating ROS2 nodes.
In fact, this code simply creates a new node class to implement our desired functionality.

An instance of this node class will be assigned the name `object_detector_node`, and it will subscribe to the `camera_topic` topic and publish `Bool` type messages to the `object_detected` topic.
The `camera_topic` topic contains the video stream received by the robot's camera sensor, and we will use the `image_publisher` package and the images prepared earlier to simulate this video stream.
The `object_detected` topic will contain our detection results for use by subsequent nodes in the robot's logic chain.
If we detect a pineapple, we will publish a `True` message; otherwise, we will publish a `False` message.

Now, let us focus on some new details in this new node class.

First, we imported the `cv_bridge.CvBridge` class.
This class is a utility class within the ROS2 framework that primarily helps us convert images between `opencv`/`numpy` format and ROS2's own `sensor_msgs.msg.Image` message format.
In our new node class, we can see its specific usage (i.e., `self.cv_bridge = CvBridge()` and `img = self.cv_bridge.imgmsg_to_cv2(msg)`).

Then, in the new node class `ObjectDetectorNode`, we used the class member variable `PINEAPPLE_CLASS_ID` to store the class ID of the object we want to recognize in ImageNet. Here, `953` is the specific class ID for pineapple in ImageNet.

After that, we instantiated a pre-trained AlexNet through PyTorch and set it to `eval` mode.
At the same time, we declared the `detection_class_id` parameter to make it convenient to modify the class ID of the object to be recognized at runtime (though this is not commonly used).

Finally, in the callback function `camera_callback` for the `camera_topic` topic, we convert the received `Image` type message to `numpy` format, then call AlexNet for object recognition, and finally publish the recognition result as a `Bool` to the `object_detected` topic while logging the result.

With this, a node class that uses PyTorch and AlexNet to recognize whether there is a pineapple in the camera view is complete.

### Running and Testing

Now, let us try running our newly written node class and test whether it works correctly using pineapple and apple images.

First, let us build this newly written Python package.

```shell
cd openmlsys-ros2
colcon build --symlink-install
```

After successful compilation, we can open a new terminal window and run the following command to start a node class instance.
Remember, you may need to first run `source install/local_setup.zsh` to source our own ROS2 project.

```shell
ros2 run object_detector object_detector_node --ros-args -r camera_topic:=image_raw
```

If you encounter issues like `ModuleNotFoundError: No module named 'cv2'`, it means the ROS2 command did not successfully find the libraries in your Python virtual environment. In this case, you can try running the following command after entering your virtual environment.

```shell
PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH" ros2 run object_detector object_detector_node --ros-args -r camera_topic:=image_raw
```

The purpose of the `PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH"` prefix is to add the libraries from your current Python environment to `PYTHONPATH`, so that the Python used by the ROS2 command can find the Python libraries in your current Python environment (i.e., the Python virtual environment corresponding to the ROS2 project).

When this ROS2 command runs successfully, you should see this message: `[INFO] [1655172977.491378700] [object_detector_node]: Detector node is ready.`.

Additionally, in this ROS2 command, we used the `--ros-args -r camera_topic:=image_raw` series of arguments.
These arguments tell ROS2 to remap the `camera_topic` topic used by our new node class to the `image_raw` topic.
This way, all places where our new node class uses the `camera_topic` topic will actually be using the `image_raw` topic.
The benefit of topic name remapping is decoupling.
For each new ROS2 package or each new node class, we can freely name the topics we want to use, and then when it needs to work together with other components, we simply use remapping to connect the different topic names used by different components, achieving normal data flow between the two components.
This is actually a very practical feature of the ROS2 framework.

If you want to learn more about remapping details, you can read [this official introduction](https://design.ros2.org/articles/static_remapping.html).

After we successfully start the new node, let us run the following command in a new terminal window to test whether it can detect pineapples. Again, you may need to first run `source install/local_setup.zsh` to source our own ROS2 project.

```shell
PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH" ros2 run image_publisher image_publisher_node data/ros-pineapple.jpg --ros-args -p publish_rate:=1.0
```

The above command will use the `image_publisher` package and its node to publish the previously prepared pineapple image to the `image_raw` topic at a frequency of 1Hz.
When this `image_publisher_node` node runs successfully, we should see messages similar to `[INFO] [1655174212.930385900] [object_detector_node]: Detected: "True", target class id: 953` in the terminal window where the `object_detector_node` node is running, confirming that our node class can detect pineapples.

Next, let us stop the `image_publisher_node` node by pressing `Ctrl+C` in its window, and then use the following command to publish the prepared apple image.

```shell
PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH" ros2 run image_publisher image_publisher_node data/ros-apple.jpg --ros-args -p publish_rate:=1.0
```

Now, in the terminal window where the `object_detector_node` node is running, we should see messages similar to `[INFO] [1655171989.912783400] [object_detector_node]: Detected: "False", target class id: 953`, confirming that our node class will not mistake an apple for a pineapple.

### Summary

Congratulations, you have successfully learned how to use Python libraries outside the ROS2 framework in a ROS2 project!
If you use a Python virtual environment, you may need to additionally set the `PYTHONPATH` environment variable.
Additionally, topic name remapping (Name Remapping) is a very useful ROS2 feature.
You will likely use it frequently in future projects.
