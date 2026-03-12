## Planning System Case Study

In the previous section, we gained a preliminary understanding of the robot's planning system.
In this section, we will demonstrate how to combine ROS2 and the machine learning framework scikit-learn to implement a basic function of our envisioned planning system through a simple case study.
We will use a method and structure similar to the [Perception System Case Study](./perception_code_ex.md) section to present this chapter.

### Case Background

Suppose we want to design a gardener robot to tend iris flowers for a certain garden.
"Coincidentally," this small garden happens to contain only the three species of iris from the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), and someone has already created a "magical" ROS2 perception component that automatically detects the target iris flower's Sepal Length, Sepal Width, Petal Length, and Petal Width (the four input dimensions required by the Iris dataset).
At the same time, due to the robot's performance limitations, we cannot use relatively complex models (such as neural networks).
In this situation, we can try using a classical machine learning model, such as a decision tree, to accept the results from the perception component and identify the iris species, then use a mapping table to look up what behavior to plan for the robot to execute.
When the season or circumstances change, the garden's technical team can update the mapping table to modify the robot's planning system logic.

Of course, the case background and solution described above are "unrealistic" examples designed to create a simple case study.
Cases encountered in real-world projects will be much more complex.
Nevertheless, we still hope that such a simple case study can bring some value to readers.

Let us return to the solution we just introduced.
In the previous perception system case study, we chose to use a ROS2 node class to handle the perception task.
This is because the robot continuously receives signals from sensors, and we want to process as many received signals as possible.
For this chapter's case study, since we do not necessarily need to plan continuously, and we expect a result from each planning request, using a ROS2 service may be a better choice.

Similar to previous case studies, the code used in this section can be found in the `src/action_decider` folder of the book's related [ROS2 case study code repository](https://github.com/openmlsys/openmlsys-ros2).

### Project Setup

Let us continue using the ROS2 project framework we have already set up.
Similar to the perception system case study, we only need to add a new ROS2 Python package to implement the desired functionality.
Therefore, let us go back to the `src` directory and create this Python package.

```shell
cd openmlsys-ros2/src
ros2 pkg create --build-type ament_python --node-name action_decider_node action_decider --dependencies rclpy std_msgs scikit-learn my_interfaces
```

We added `my_interfaces` as a dependency because we need to create the corresponding message type interface for the new ROS2 service.

After creating the Python package, do not forget to update the `version`, `maintainer`, `maintainer_email`, `description`, and `license` fields in `package.xml` and `setup.py`.

Next, let us install `scikit-learn` in the ROS2 project's Python virtual environment.
For example, users using `pipenv` might run the command `pipenv install scikit-learn`.

### Adding Message Type Interface

The new ROS2 service we are about to write needs its own service message interface.
Let us use the existing `my_interfaces` package to house this new interface.

First, let us create a new file named `IrisData.srv` in `openmlsys-ros2/src/my_interfaces/srv` and populate it with the following content.

```text
float32 sepal_length
float32 sepal_width
float32 petal_length
float32 petal_width
---
string action
```

We can see that the new ROS2 service will accept four floating-point values as input.
These four floating-point values represent the sepal length and width, and the petal length and width of the iris flower.
When planning is complete, the service will return a string.
This string will be the name of the action the robot needs to execute.

We also need to add a new line at the appropriate location (in the parameter section of the `rosidl_generate_interfaces` function) in the `CMakeLists.txt` file of the `my_interfaces` package:

```cmake
"srv/IrisData.srv"
```

Finally, do not forget to run `colcon build --packages-select my_interfaces` in the root directory of the ROS2 project to recompile the `my_interfaces` package.

### Adding Code

The command we used to create the Python package should have already created the file `src/action_decider/action_decider/action_decider_node.py`. Now let us replace the existing content of this file with the following content.

```Python
import os
import pickle

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from my_interfaces.srv import IrisData

from sklearn.datasets import load_iris
from sklearn import tree


def main(args=None):
    rclpy.init(args=args)
    action_decider_service = ActionDeciderService()
    rclpy.spin(action_decider_service)
    action_decider_service.destroy_node()
    rclpy.shutdown()


class ActionDeciderService(Node):

    IRIS_CLASSES = ['setosa', 'versicolor', 'virginica']

    IRIS_ACTION_MAP = {
        'setosa': 'fertilise',
        'versicolor': 'idle',
        'virginica': 'prune',
    }

    DEFAULT_MODEL_PATH = f'{os.path.dirname(__file__)}/../../../data/iris_model.pickle'

    def get_iris_classifier(self, model_path):
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as model_file:
                return pickle.load(model_file)
        self.get_logger().info(f"Cannot find trained model at '{model_path}', will train a new model.")
        iris = load_iris()
        X, y = iris.data, iris.target
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
        with open(model_path, 'wb') as model_file:
            pickle.dump(clf, model_file)
        return clf

    def __init__(self):
        super().__init__('iris_action_decider_service')
        self.srv = self.create_service(IrisData, 'iris_action_decider', self.decide_iris_action_callback)
        self.iris_classifier = self.get_iris_classifier(self.DEFAULT_MODEL_PATH)
        self.get_logger().info('Iris action decider service is ready.')

    def decide_iris_action_callback(self, request, response):
        iris_data = [request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]
        iris_class_idx = self.iris_classifier.predict([iris_data])[0]
        iris_class = self.IRIS_CLASSES[iris_class_idx]
        response.action = self.IRIS_ACTION_MAP[iris_class]
        self.get_logger().info(
            f'Incoming request\nsepal_length: {request.sepal_length}\nsepal_width: {request.sepal_width}'
            f'\npetal_length: {request.petal_length}\npetal_width: {request.petal_width}'
            f'\niris class: {iris_class}'
            f'\ndecided action: {response.action}'
        )

        return response


if __name__ == '__main__':
    main()
```

Observant readers may have already noticed that this code is very similar to our previous code for creating a ROS2 service server node class.
In fact, this code uses the same service server node class framework with a new service to implement the desired functionality.

An instance of this service server node class will be assigned the name `iris_action_decider_service`. It will provide a service named `iris_action_decider`, and this service expects `IrisData` format service requests (i.e., the request portion of the message type interface we defined earlier).
When the service computation is complete, it will return the result to the requester.
This result is the name of the planned action, packaged into the `IrisData` format service response (i.e., the response portion of the message type interface we defined earlier).

Now, let us focus on some new details in this new node class.

First, in the new service server node class `ActionDeciderService`, we declared three class member variables: `IRIS_CLASSES`, `IRIS_ACTION_MAP`, and `DEFAULT_MODEL_PATH`.
They represent the iris class labels, the mapping table from iris classes to robot action names, and the default path for storing the trained decision tree model, respectively.

When our service server node class initializes, it calls `get_iris_classifier()` to load the trained decision tree model.
If the model file is missing, it will retrain a model and save it.
Here we placed the model training code within the same node.
In practice, for large projects or large models, we can separate model training and model usage into different components, and they may run at different times.

When the service callback function `decide_iris_action_callback()` is called, the service will use the trained model and the received iris information to predict the iris class, then look up the mapping table to determine the action the robot needs to execute. Finally, the service returns the result and logs it.

With this, a simple "toy-level" planning component using scikit-learn and decision trees is complete.

### Running and Testing

Now, let us try running the newly written service server node class and test whether it works correctly.

First, let us build this newly written Python package.

```shell
cd openmlsys-ros2
colcon build --symlink-install
```

After successful compilation, we can open a new terminal window and run the following command to start a node class instance.
Remember, you may need to first run `source install/local_setup.zsh` to source our own ROS2 project.

```shell
ros2 run action_decider action_decider_node
```

If you are using a Python virtual environment, you can try the following command instead. The specific reason behind this has been described in the previous case study section.

```shell
PYTHONPATH="$(dirname $(which python))/../lib/python3.8/site-packages:$PYTHONPATH" ros2 run action_decider action_decider_node
```

When this ROS2 command runs successfully, you should see this message: `[INFO] [1655253519.693893500] [iris_action_decider_service]: Iris action decider service is ready.`.

After we successfully start the new service server node, let us run the following command in a new terminal window to test whether the new service works correctly. Again, you may need to first run `source install/local_setup.zsh` to source our own ROS2 project.

```shell
ros2 service call /iris_action_decider my_interfaces/srv/IrisData "{sepal_length: 1.0, sepal_width: 2.0, petal_length: 3.0, petal_width: 4.0}"
```

Here, the `ros2 service call` command is specifically used to call a ROS2 service from the command line. The service request data should be in stringified YAML format. More information about this command can be found by running `ros2 service call -h`.

If everything goes well, shortly after executing the command, you should see a message similar to: `response: my_interfaces.srv.IrisData_Response(action='prune')` in the new window.

### Summary

Congratulations, you have successfully learned how to use libraries like scikit-learn in a ROS2 project and train a model!
