## Neural Networks

### Perceptron
![A neuron with three inputs and a single output](../img/ch_basic/single_neuron2.png)
:width:`600px`
:label:`single_neuron`

 :numref:`single_neuron` shows an example of a neuron, where the input data $x$ is weighted and summed according to the weights $w$ on the connections to produce the output $z$. We call such a model a **perceptron**.
 Since there is only one layer of neural connections between input and output, this model is also called a single-layer perceptron. The computation of the model in :numref:`single_neuron` can be written as: $z = w_{1}x_{1}+ w_{2}x_{2} + w_{3}x_{3}$.

When the input data is represented as a column vector ${x}=[x_1,x_2,x_3]^T$ and the model weights are represented as a row vector ${w}=[w_1,w_2,w_3]$, the output scalar $z$ can be written as:

$$z =
\begin{bmatrix}
w_1,w_2,w_3\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
={w}{x}$$

We can use the output scalar $z$ as a weighted combination of the inputs to accomplish specific tasks.
For example, we can classify "good apples" and "bad apples," where $x_1,x_2,x_3$ represent three different features: 1) degree of redness, 2) presence of holes, and 3) size. If the size of the apple has no effect on this judgment, the corresponding weight would be zero.
Training this neural network essentially means selecting appropriate weights to accomplish our task. For instance, we can choose appropriate weights such that when $z$ is less than or equal to $0$, it represents a "bad apple," and when $z$ is greater than $0$, it represents a "good apple."
The final classification output label $y$ is as follows, where $1$ represents good and $0$ represents bad. Since there is only one layer between the input and output of this neuron, it can be called a single-layer neural network.

$$
y =
\begin{cases}
1 &  z>0 \\
0 & z \leq 0 \\
\end{cases}$$

### Decision Boundary vs. Bias

By selecting appropriate weights and classifying input data based on whether $z$ is greater or less than $0$, we can obtain a **decision boundary** in the data space. As shown in :numref:`single_neuron_decision_boundary2`, using the neuron output $z=0$ as the decision boundary for the output label $y$,
without bias the decision boundary must pass through the origin. If the data sample points are not separated by the origin, classification errors will occur.
To solve this problem, a **bias** can be added to the neuron. :numref:`single_neuron_bias2`
shows a neuron model with bias $b$, which can be expressed by :eqref:`singleneuron_bias`:
$$z = w_{1}x_{1}+ w_{2}x_{2}+ w_{3}x_{3} + b$$
:eqlabel:`singleneuron_bias`

![Decision boundaries with two inputs (left) and three inputs (right). Different shaped points represent different classes of data, and we need to find $z=0$ as the decision boundary to separate the different data points. With two inputs, the decision boundary is a line; with three inputs, the decision boundary is a plane; with higher-dimensional inputs, the decision boundary is called a **hyperplane**.
Left: $z=w_{1}x_{1}+w_{2}x_{2}+b$. Right: $z=w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+b$. Without bias, the decision boundary must pass through the origin, so it cannot separate the data samples of different classes.](../img/ch_basic/single_neuron_decision_boundary2.png)
:width:`600px`
:label:`single_neuron_decision_boundary2`

![A single-layer neural network with bias](../img/ch_basic/single_neuron_bias2.png)
:width:`600px`
:label:`single_neuron_bias2`

With bias, the decision boundary (line, plane, or hyperplane) does not have to pass through the origin, thus enabling better classification of samples.
More precisely, the decision boundary separates the sample data into two different classes, and this boundary is
$\{x_1, x_2, x_3 | w_{1}x_{1}+ w_{2}x_{2}+ w_{3}x_{3} + b = 0\}$.

### Logistic Regression

The input-output relationship of the above neuron is linear. To provide nonlinear data representation capability, an **activation function** can be applied to the neuron output. The most common activation functions include Sigmoid, Tanh, ReLU, and Softmax.
For example, the above neuron uses $z=0$ as the boundary for classification tasks. Can we instead have the neuron output a probability? For instance, outputting values between $0$ and $1$, where $1$ means the input data belongs to a certain class with $100\%$ probability.
To make the neuron output values between $0$ and $1$, we can apply the logistic function **Sigmoid** to $z$,
as shown in :eqref:`sigmoid`. Sigmoid constrains values between 0 and 1, and a simple threshold (e.g., 0.5) can be used to determine whether the final output label belongs to a certain class. This method is called **logistic regression**.

$$a = f({z}) = \frac{1}{1+{\rm e}^{-{z}}}$$
:eqlabel:`sigmoid`

### Multiple Neurons

![Multiple neurons](../img/ch_basic/two_neurons2.png)
:width:`600px`
:label:`two_neurons2`

The above network has only one output. With multiple neurons together, we can have multiple outputs. :numref:`two_neurons2` shows a network with two outputs, where each output is connected to all inputs. This is also called a **fully-connected (FC) layer**,
which can be expressed by the following equation :eqref:`fc_cal`.

$$z_{1} &= w_{11}x_{1} + w_{12}x_{2} + w_{13}x_{3} + b_1 \notag \\ z_{2} &= w_{21}x_{1} + w_{22}x_{2} + w_{23}x_{3} + b_2$$
:eqlabel:`fc_cal`

The following expression shows the matrix form of the computation:

$$
{z} =
\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}
=
\begin{bmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23}\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
+
\begin{bmatrix}
b_1 \\ b_2
\end{bmatrix}
= {W}{x} + {b}$$


A network with multiple outputs can solve multi-class classification problems. For example, with 10 numerical outputs, each value represents the probability of a particular class, with each output between $0$ and $1$, and the sum of all 10 outputs equal to $1$.
This can be achieved using the **Softmax** function shown in :eqref:`e_softmax`, where $K$ is the number of outputs:

$$f({z})_{i} = \frac{{\rm e}^{z_{i}}}{\sum_{k=1}^{K}{\rm e}^{z_{k}}}$$
:eqlabel:`e_softmax`

### Multi-Layer Perceptron

![Multi-layer perceptron example. $a^l_i$ represents the value after the neuron output $z$ passes through the activation function, where $l$ denotes the layer index ($L$ denotes the output layer), and $i$ denotes the output index](../img/ch_basic/mlp2.png)

**Multi-Layer Perceptron** (MLP) :cite:`rosenblatt1958perceptron` enhances the network's representation capability by stacking multiple fully-connected layers. Compared to single-layer networks, the multi-layer perceptron has many intermediate layer outputs that are not exposed to the final output; these layers are called **hidden layers**. The network in this example can be implemented through the following cascaded matrix operations, where $W^l$ and $b^l$ represent the weight matrices and biases of different layers, $l$ denotes the layer index, and $L$ denotes the output layer.

$${z} = f({W^L}f({W^3}f({W^2}f({W^1}{x} + {b^1}) + {b^2}) + {b^3}) + {b^L})$$

In the deep learning era, network models are essentially composed of multiple layers of neural network layers connected together. Input data passes through multiple layers of feature extraction, learning **feature vectors** at different levels of abstraction. Below we introduce some other commonly used neural network layers.

### Convolutional Networks

![Convolution operation example. The input is a three-channel data of size $4 \times 4 \times 3$ (height $\times$ width $\times$ channels). To perform convolution on each channel, the convolution kernel must also have three channels. A single convolution kernel has size $3 \times 3 \times 3 \times 1$ (height $\times$ width $\times$ input channels $\times$ output channels (number of kernels)). The number of convolution kernels determines the number of output **feature maps**. In this example, since there is only one convolution kernel, the output has 1 channel with height and width of 2. We call such high-dimensional input data **tensors**, such as RGB images, videos, outputs from previous convolutional layers, etc.](../img/ch_basic/conv_computation_v4.png)
:width:`600px`
:label:`conv_computation_v4`

**Convolutional Neural Network** (CNN) :cite:`lecun1989backpropagation` consists of multiple **convolutional layers** and is commonly used in computer vision tasks :cite:`krizhevsky2012imagenet,he2016deep`.
 :numref:`conv_computation_v4` describes an example of a convolution operation.
Based on the properties of convolution, we can observe two facts: 1) the number of channels in a convolution kernel equals the number of input channels; 2) the number of output channels equals the number of convolution kernels.

In the example of :numref:`conv_computation_v4`, the convolution kernel slides by one unit at a time to perform the convolution operation; we say its **stride** is 1. Additionally, if we want the edge values of the input to also be taken into account, we need to perform **zero padding** on the edges. In the example of :numref:`conv_computation_v4`, if each channel of the input is padded with a ring of zeros on all four sides, the output size would be $4\times 4\times 1$. The number of padding rings depends on the kernel size---larger kernels require more padding.

To perform feature extraction on input image data, the number of convolution kernels is typically greater than the number of input channels, which means the output data contains many more values and the computation increases. However, features of adjacent pixels in image data are often similar, so we can perform aggregation operations on adjacent output features. **Pooling layers** serve this purpose, and we typically use two pooling methods: Max Pooling and Mean Pooling. As shown in :numref:`pooling_v3`, assuming a pooling kernel of size $2\times2$, an input of $4\times4$, and a stride of 2 (with stride 1, the output equals the input), the output is $2\times2$.

![$2 \times 2$
max pooling and mean pooling examples, with stride 2 and input size $4 \times 4$](../img/ch_basic/pooling_v3.png)
:width:`600px`
:label:`pooling_v3`

Both convolutional layers and fully-connected layers are commonly used. However, when the input is high-dimensional image data, convolutional layers require far fewer parameters than fully-connected layers. The operations in convolutional layers are similar to those in fully-connected layers---the former is based on high-dimensional tensor operations, while the latter is based on two-dimensional matrix operations.

### Sequential Models

In real life, besides images, there is a large amount of time series data, such as videos, stock prices, and so on. **Recurrent Neural Networks** (RNN) :cite:`rumelhart1986learning` are a type of deep learning model architecture designed for processing sequential data. Sequential data is a series of continuous data $\{x_1, x_2, \dots, x_n\}$, where each $x$ might represent a word in a sentence, for example.

To receive a continuous sequence of inputs, as shown in :numref:`rnn_simple_cell2`, the vanilla recurrent neural network uses a recurrent cell as the computation unit, with a hidden state to store information from past inputs. Specifically, for each input data $x$ to the model, according to equation :eqref:`aligned`, the recurrent cell repeatedly computes new hidden states to record information from current and past inputs. The new hidden state is then used in the computation of the next cell.

$${h}_t = {W}[{x}_t; {h}_{t-1}] + {b}$$
:eqlabel:`aligned`

![Vanilla recurrent neural network. At each computation step, the recurrent cell computes the current hidden state ${h}_t$ from the previous hidden state ${h}_{t-1}$ and the current input ${x}_t$.](../img/ch_basic/rnn_simple_cell2.png)
:width:`600px`
:label:`rnn_simple_cell2`

However, this simple vanilla recurrent neural network suffers from a severe information forgetting problem. For example, if the input is "I am Chinese, my native language is ___," the hidden state remembers the information about "Chinese," enabling the network to predict the word "Chinese (language)" at the end. But when the sentence is very long, the hidden state may not remember information from too long ago. For instance, "I am Chinese, I went to study in the UK, then worked in France, my native language is ___"---at this point, the information about "Chinese" in the final hidden state may have been forgotten due to multiple updates.
To address this problem, various improved methods have been proposed, the most famous being Long Short-Term Memory (LSTM) :cite:`Hochreiter1997lstm`. There are many more sequential models, such as the Transformer :cite:`vaswani2017attention` that emerged in recent years.