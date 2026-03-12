## Computation Scheduling and Execution

After operator selection and memory allocation, computation tasks can be scheduled and executed on hardware through the runtime. Depending on whether operators are compiled into a computational graph, computation scheduling can be divided into two approaches: single-operator scheduling and graph scheduling. For example, MindSpore provides the PyNative mode and Graph mode respectively. Furthermore, depending on the hardware capabilities, the execution of computational graphs can be divided into two modes: interactive execution, where operators are dispatched and executed one by one, and sink execution, where the entire computational graph or partial subgraphs are dispatched to the hardware at once.

### Single-Operator Scheduling

Single-operator scheduling, as opposed to graph-based scheduling, means that operators contained in algorithms or models are scheduled and executed one by one through the Python runtime. Examples include PyTorch's default execution mode, TensorFlow's eager mode, and MindSpore's PyNative mode. Taking MindSpore as an example, the code is shown below.

```python
import mindspore.nn as nn
from mindspore import context

class Computation(nn.Cell):
   def construct(self, x, y):
     m = x * y
     n = x - y
     print(m)
     z = m + n
     return z

compute = Computation()
c = compute(1, 2)
print(c)
```

The above script defines all computation logic in the `construct` method of the `Computation` class. Since single-operator execution mode is preset in the context at the beginning of the script, the computations in `construct` will be called and executed line by line through the Python runtime, and `print` commands can be inserted at any position in the code to print intermediate computation results.

The call chain for single-operator execution is shown in :numref:`single_op_exec`. After an operator is triggered for execution on the Python side, it goes through the machine learning framework initialization, which determines information including the operator's precision, input and output types and sizes, and the corresponding hardware device. Then the framework allocates the memory required for computation, and finally hands it over to the specific hardware computing device to complete the execution.

![Single-Operator Execution](../img/ch05/single_op_exec.PNG)
:width:`800px`
:label:`single_op_exec`

The advantage of single-operator scheduling lies in its flexibility. Since operators are directly scheduled through the Python runtime, it can express arbitrarily complex computation logic, especially in scenarios requiring complex control flow and Python native data structures to implement complex algorithms. Additionally, single-operator scheduling is very convenient for debugging program correctness, as developers can print any variable that needs to be debugged during code execution. Finally, by driving operators through the Python runtime, computation tasks can be completed in coordination with Python's vast and rich ecosystem of libraries.

### Graph Scheduling

Although single-operator scheduling has the advantages described above, its disadvantages are also obvious. On one hand, it is difficult to optimize computation performance, because without global information from the computational graph, single-operator execution cannot perform optimizations such as operator fusion and algebraic simplification based on context. On the other hand, due to the lack of topological relationships in the computation, the entire computation can only be scheduled and executed serially, meaning that parallel computation cannot be achieved through the runtime. For example, the computation logic of the above sample code can be expressed as shown in :numref:`graph_exec`. From this computational graph, we can see that there is no dependency between the multiplication and subtraction operations, so these two computations can be executed in parallel. Such parallel execution information can only be analyzed after the computation is expressed as a computational graph, which is one of the advantages of graph scheduling over single-operator scheduling.

![Computational Graph](../img/ch05/graph_exec.png)
:width:`800px`
:label:`graph_exec`

Now let us introduce the scheduling methods for computational graphs. In a typical heterogeneous computing environment, there are multiple types of computing devices such as CPUs, GPUs, and NPUs. Therefore, a computational graph can be composed of operators running on different devices, forming a heterogeneous computational graph. :numref:`computation_graph` shows a typical computational graph involving heterogeneous hardware.

![Heterogeneous Hardware Computational Graph](../img/ch05/computation_graph.png)
:width:`800px`
:label:`computation_graph`

The computational graph described above consists of operators corresponding to the following types of heterogeneous hardware:

-   **CPU Operators**: Operators written in C++ and executed on the host via the CPU. The performance of CPU computation depends on whether the multi-core computing capability of the CPU can be fully utilized.

-   **GPU Operators**: Taking NVIDIA GPU chips as an example, GPU Kernels are dispatched one by one from the host side to the GPU device, where the GPU chip executes the operator's computation logic. Due to the large number of parallel execution units on the chip, it can provide powerful acceleration capabilities for highly parallel algorithms.

-   **NPU Operators**: Taking Huawei Ascend chips as an example, Ascend is a highly integrated SoC chip. The advantage of NPUs is their support for sinking part of or the entire computational graph into the chip to complete computation. During computation, there is no interaction with the host, resulting in higher computational performance.

-   **Python Operators**: Similar to CPU operators in execution mode, both are executed by the host's CPU. The difference is that the computation logic is interpreted and executed by the Python runtime through the Python interpreter.

The prerequisite for correctly expressing a heterogeneous computational graph is to accurately identify the device on which each operator executes. For example, the CPU, GPU, and Ascend Kernels identified in the heterogeneous computational graph :numref:`computation_graph`, as well as the Python Kernels marked to be executed by the Python runtime. Mainstream frameworks all provide the capability to specify the device on which an operator runs. Taking MindSpore as an example, a simple heterogeneous computation code is shown below.

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops.operations as ops
from mindspore.common.api import jit

# Create operators and specify the hardware device for execution
add = ops.Add().add_prim_attr('primitive_target', 'CPU')
sub = ops.Sub().add_prim_attr('primitive_target', 'GPU')

# Specify execution in static computational graph mode
@jit
def compute(x, y, z):
    r = add(x, y)
    return sub(r, z)

# Create arguments
x = Tensor(np.ones([2, 2]).astype(np.float32))
y = Tensor(np.ones([2, 2]).astype(np.float32))
z = Tensor(np.ones([2, 2]).astype(np.float32))

# Execute computation
output = compute(x, y, z)
```

The above code snippet completes the computation logic of x + y - z, where the Add operator is set to execute on the CPU and the Sub operator is set to execute on the GPU, forming CPU-GPU collaborative heterogeneous computation. Through a similar tagging mechanism, arbitrarily complex multi-hardware collaborative heterogeneous computation can be expressed.
Another relatively special type of heterogeneity involves Python operators. The advantages of Python lie in its flexibility of expression, development efficiency, and rich surrounding ecosystem. Therefore, introducing Python operators into the computational graph to collaborate with operators on other heterogeneous hardware greatly enhances computation flexibility. Unlike the heterogeneity where CPU and GPU execute on different devices, Python operators and CPU operators implemented in C++ are both executed by the host-side CPU cores. The difference is that Python operators are described through a unified computational graph and therefore also need to be triggered for execution in the backend runtime. To express Python operators in the computational graph, the framework needs to provide corresponding support.

After marking the devices corresponding to operators in the computational graph, the graph is ready to be scheduled and executed. Depending on hardware capabilities, the execution of heterogeneous computational graphs can be divided into three modes: operator-by-operator interactive execution, whole-graph sink execution, and subgraph sink execution. Interactive execution is mainly for CPU and GPU scenarios, where operators in the computational graph are scheduled and executed one by one according to the dependency relationships of inputs and outputs. Whole-graph sink execution is mainly for NPU chips, whose main advantage is the ability to dispatch the entire neural network's computational graph to the device at once, independently completing the scheduling and execution of all operators in the graph without relying on the host's CPU capability, reducing the number of interactions between host and chip, and improving computational efficiency and performance through the NPU's tensor acceleration capability. Subgraph sink execution combines the previous two execution modes. Due to the flexibility of computational graph expression itself, whole-graph sink execution on NPU chips may not achieve optimal efficiency for complex scenarios. Therefore, parts with low execution efficiency on NPU chips can be separated and handed over to devices with higher execution efficiency such as CPUs or GPUs, while subgraphs more suitable for NPU computation are sunk to the NPU for computation, thus balancing both performance and flexibility.

The above heterogeneous computational graph can serve two purposes. The first is heterogeneous hardware acceleration, placing specific computations on suitable hardware for execution. The second is achieving concurrent execution between operators. From the computational graph, we can see that there is no dependency between kernel_1 and kernel_2, nor between kernel_3 and kernel_4. Therefore, these two pairs of CPU and GPU operators can logically be invoked concurrently by the framework. However, kernel_5 depends on the outputs of kernel_3 and kernel_4 as its inputs, so kernel_5 needs to wait for kernel_3 and kernel_4 to complete before being triggered for execution.

Although concurrency relationships between operators can be fully expressed on the computational graph, in practice, some unexpected side effects may arise due to concurrency, as shown in the following code:

```python
import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.ops.operations as ops
from mindspore.common.api import jit

# Define global variables
x = Parameter(Tensor([1.0], ms.float32), name="x")
y = Tensor([0.2], ms.float32)
z = Tensor([0.3], ms.float32)

# Specify execution in static computational graph mode
@jit
def compute(y, z):
    ops.Assign()(x, y)
    ops.Assign()(x, z)
    r = ops.Sub()(x, y)
    return r

compute(y, z)
```

The above code expresses the following computation logic:

```text
x = y
x = z
x = x - y
```

This simple computation logic, when translated to the computational graph, can be represented as shown in :numref:`side_effect_1`.

![Concurrent Operator Execution](../img/ch05/side_effect_1.png)
:width:`800px`
:label:`side_effect_1`

There are no dependencies among the three computations shown in the code, so these three operators can logically be executed concurrently on the computational graph. However, based on the code semantics, it is obvious that the program needs to be executed sequentially. The issue introduced here is called a side effect, which refers to the behavior of modifying state variables defined outside the function. Due to the introduction of side effects, incorrect concurrency relationships occur. One solution is to add dependencies between operators during the computational graph compilation phase to convert concurrent execution logic into sequential execution logic. The transformed computational graph is shown in :numref:`side_effect_2`.

![Eliminating Side Effects](../img/ch05/side_effect_2.png)
:width:`800px`
:label:`side_effect_2`


The dashed arrows in the figure represent the dependency relationships between operators. After adding dependency relationships, the operators will execute serially in the order of Assign_1, Assign_2, Sub_1, which is consistent with the original code semantics.

### Interactive Execution

As described above, in interactive execution mode, the framework's runtime dispatches operators to the hardware for execution one by one according to the dependency relationships of operators in the computational graph, following a certain execution order (e.g., breadth-first order). To aid understanding and comparison, we first introduce the execution method for non-heterogeneous computational graphs (where all operators in the graph run on the same type of device), as heterogeneous computational graph execution is built upon non-heterogeneous graphs.

1. Execution of Non-Heterogeneous Computational Graphs

![Non-Heterogeneous Computational Graph](../img/ch05/graph_exec_1.png)
:width:`800px`
:label:`graph_exec_1`

As shown in :numref:`graph_exec_1`, this is a non-heterogeneous computational graph where all Kernels are GPU operators. The execution methods are generally divided into serial execution and parallel execution:

![Serial Execution](../img/ch05/graph_exec_2.png)
:width:`800px`
:label:`graph_exec_2`

![Parallel Execution](../img/ch05/graph_exec_3.png)
:width:`800px`
:label:`graph_exec_3`

-   **Serial Execution**: The computational graph is unfolded into an execution sequence, and operators are executed serially one by one according to the execution order, as shown in :numref:`graph_exec_2`. Its characteristics include a fixed execution order, single-threaded execution, and relatively low system resource requirements.

-   **Parallel Execution**: The computational graph is unfolded according to the dependency relationships between operators. Operators with dependencies maintain their execution order through input dependencies, while operators without dependencies can be executed in parallel, as shown in :numref:`graph_exec_3`. Kernel_1 and Kernel_2 have no dependencies and can execute in parallel, and Kernel_3 and Kernel_4 have no dependencies and can execute in parallel. Its characteristics include a non-fixed execution order (the order of operators executed in each round is likely to differ), multi-threaded execution, and relatively high system resource requirements.

Serial execution and parallel execution each have their advantages and disadvantages, summarized in :numref:`serial_vs_parallel`.

:Comparison of Serial Execution and Parallel Execution

| Execution Method     | Serial Execution |  Parallel Execution |
|--------------|----------|------|
|Operator Execution Order    | Fixed  |    Non-fixed |
|Operator Execution Thread    |Single-threaded  |   Multi-threaded |
|Required Execution Resources    | Lower  |     Higher |
:label:`serial_vs_parallel`

2. Execution of Heterogeneous Computational Graphs

![Heterogeneous Computational Graph](../img/ch05/graph_exec_4.png)
:width:`800px`
:label:`graph_exec_4`

As shown in :numref:`graph_exec_4`, this is a heterogeneous computational graph, where Kernel_1, Kernel_2, Kernel_5, and Kernel_9 are CPU operators, Kernel_6 is a Python operator (also executed on the CPU), Kernel_3 and Kernel_4 are GPU operators, and Kernel_7 and Kernel_8 are GPU operators.
Generally, computational graph optimizations are implemented based on non-heterogeneous computational graphs, requiring all operators in the graph to be on the same device to facilitate optimizations such as operator fusion and replacement. Therefore, a heterogeneous computational graph needs to be partitioned into multiple non-heterogeneous computational graphs. The partitioning can be quite flexible, with various partitioning rules defined. Generally, partitioning rules that produce as few subgraphs as possible are used, placing as many operators on the same device into one subgraph as possible. As shown in :numref:`graph_exec_5`, five subgraphs are produced: Graph_1\_CPU, Graph_2\_GPU, Graph_3\_CPU, Graph_4\_Ascend, and Graph_5\_CPU.

![Heterogeneous Computational Graph Partitioning](../img/ch05/graph_exec_5.png)
:width:`800px`
:label:`graph_exec_5`

After partitioning a heterogeneous computational graph into multiple subgraphs, the execution methods are generally divided into subgraph partitioned execution and subgraph merged execution:

-   **Subgraph Partitioned Execution**: The partitioned subgraphs are executed separately, i.e., one subgraph finishes execution before the next one starts, as shown in :numref:`graph_exec_6`. The output data of the previous subgraph is transferred to the input of the next subgraph, and the next subgraph needs to copy the input data to its own device memory. For example, Graph_2\_GPU needs to copy the output data of Graph_1\_CPU from CPU to GPU, and conversely, Graph_3\_CPU needs to copy the output data of Graph_2\_GPU from GPU to CPU. There is a certain overhead in switching execution between subgraphs.

-   **Subgraph Merged Execution**: The partitioned subgraphs are merged into a single overall DAG for execution, as shown in :numref:`graph_exec_7`. Copy operators are inserted based on operator device attributes to enable data transfer between operators on different devices, and the copy operators are also incorporated into the whole graph, forming a large unified graph for execution, reducing the overhead of switching between subgraphs.

![Subgraph Partitioning](../img/ch05/graph_exec_6.png)
:width:`800px`
:label:`graph_exec_6`

![Subgraph Merging](../img/ch05/graph_exec_7.png)
:width:`800px`
:label:`graph_exec_7`

Since subgraph merged execution can reduce the overhead of switching between subgraphs, it generally achieves higher performance. A summary comparison is shown in :numref:`partitioning_vs_merging`.

:Comparison of Subgraph Partitioning and Subgraph Merging

|     Execution Method      |   Subgraph Partitioning   |      Subgraph Merging|
|  --------------|------------------|--------------|
|   Heterogeneous Data Transfer  |   Copy between subgraphs  |   Copy between operators|
|   Additional Execution Overhead  | Subgraph switching overhead |       None|
|   Execution Concurrency Granularity  |     Subgraph-level concurrency    |   Native operator-level concurrency|
:label:`partitioning_vs_merging`


3. Execution Acceleration of Heterogeneous Computational Graphs

The previous sections described two execution methods for non-heterogeneous computational graphs and two execution methods for heterogeneous computational graphs, where heterogeneous computational graphs are built upon non-heterogeneous ones. Therefore, heterogeneous computational graphs have four possible execution methods through pairwise combination. Taking MindSpore as an example, it adopts subgraph merged parallel execution, as illustrated in :numref:`graph_exec_5`. First, executing as a single whole graph avoids the overhead of subgraph switching, and then parallel execution within the whole graph maximizes the advantage of concurrent execution, achieving optimal execution performance.

![Heterogeneous Hardware Acceleration](../img/ch05/graph_exec_8.png)
:width:`800px`
:label:`graph_exec_8`

### Sink Execution

Sink execution leverages the SoC architecture of specialized chips to schedule the entire or partial computational graph onto the chip at once to complete the computation of the full data volume. For example, with Ascend chips, a computational graph composed of multiple Ascend operators can be compiled into a Task before execution. Through the interface provided by the Ascend driver, the Task containing multiple operators is dispatched to the hardware at once for scheduling and execution. Therefore, in the above example, the Ascend operators Kernel_7 and Kernel_8 can be optimized into a subgraph Graph_4\_Ascend, which is then compiled into a Task and sunk to the Ascend for execution, as shown in :numref:`graph_exec_8`.

Sink execution achieves better overall computational performance by avoiding interactions between the host side and the device side during computation. However, sink execution also has some limitations. For example, it faces significant technical challenges in scenarios involving dynamic shape operators and complex control flow.
