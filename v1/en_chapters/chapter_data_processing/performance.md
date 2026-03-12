## Efficiency Design

In the previous section, we focused on the programming abstractions and interface design of the data module, ensuring that users can conveniently describe data processing workflows based on the APIs we provide without needing to worry too much about implementation and execution details. In this section, we will further explore the design details of key data module components such as data loading and pipeline scheduling to ensure that users can achieve optimal data processing performance. Throughout this section, we will also draw on practical experience from major existing machine learning systems to help readers deepen their understanding of these critical design approaches.

As shown in :numref:`async_data_process`, deep learning model training requires the data module to first load datasets from storage devices, perform a series of preprocessing transformations in memory, and finally send the processed data to accelerator chips for model computation. Currently, a large body of work focuses on accelerating model computation on chips through new hardware designs or operator compilation techniques, with relatively little attention paid to data processing pipeline performance issues. However, in many cases, the execution time of data preprocessing occupies a substantial proportion of the entire training task, preventing GPUs, Huawei Ascend, and other accelerators from being fully utilized. Research has shown that approximately 30% of computation time in enterprise data center workloads is spent on data preprocessing steps :cite:`murray2021tf`, and other studies have found that model training tasks on some public datasets spend 65% of their time on data preprocessing :cite:`mohan2020analyzing`. This clearly demonstrates that data module performance has a decisive impact on overall training throughput.

![Asynchronous parallel execution of data loading, preprocessing, and model computation](../img/ch07/7.3/async_data_process.png)
:width:`800px`
:label:`async_data_process`

To pursue maximum training throughput, existing systems generally choose to execute data loading, data preprocessing computation, and on-chip model computation asynchronously in parallel. These three steps form a typical producer-consumer upstream-downstream relationship. We denote the data loading rate from storage devices as F, the data preprocessing rate as P, and the on-chip data consumption rate as G. Ideally, we want G < min(F, P), so that the accelerator chip is never blocked waiting for data. However, in practice, we often encounter situations where either the data loading rate F is too low (known as I/O Bound) or the data preprocessing rate P is too low (known as CPU Bound), causing G > min(F, P) and leaving the chip underutilized. To address these critical performance issues, this section will focus on two topics:

-   How to design appropriate file formats and loading methods for the specific I/O requirements of machine learning scenarios to optimize the data loading rate F.

-   How to design parallel architectures that fully leverage the computing power of modern multi-core CPUs to improve the data processing rate P.

At the end of this section, we will also examine a challenging problem: how to leverage the computational graph compilation techniques learned in previous chapters to optimize the user's data processing computation graph, further achieving optimal data processing throughput performance. Now, let us embark on this section's brainstorming journey together.

### Efficiency of Data Loading

First, let us examine how to address the performance challenges of data loading. The first problem we face is the I/O differences caused by diverse data types and non-uniform storage formats. For example, text data may be stored in txt format, and image data may be stored in raw format or compressed formats such as JPEG. We obviously cannot design an optimal data loading scheme for every possible storage scenario. However, we can propose a unified storage format (which we call the Unirecord format) to shield against I/O differences across different data types, and then design and optimize data loading schemes based on this format. In practice, users simply need to convert their original datasets to our unified data format to benefit from efficient read performance.

![Unified data format](../img/ch07/7.3/uni_record.png)
:width:`800px`
:label:`unified_record_format`

So what other characteristics should our Unirecord have beyond unifying user storage formats? Data access in machine learning model training has the following characteristics:

-   Within each epoch, all data is traversed in a random order, with each data sample visited exactly once

-   Across all epochs, the data must be traversed in different random orders

The above access patterns require that our Unirecord storage format supports efficient random access. When our dataset can fit entirely in RAM, random access to Unirecord is not a major issue. However, when the dataset is too large and must be stored on local disks or distributed file systems, we need to design specific solutions. An intuitive approach is to divide a Unirecord file into an index block and a data block. The index block records metadata for each data sample, such as its size, offset within the file, and checksum values. The data block stores the actual data for each sample. When we need to perform random access on a Unirecord-format file, we first load the file's index block into memory (which is typically much smaller than the entire file) and build an in-memory index table for the data in the file. Then, when we need to randomly access a data sample, we first look up the sample's offset, size, and other information in the index table and read the data from disk based on this information. This loading approach satisfies our random access requirements on disk. Next, we will use the practical experience of MindRecord proposed by MindSpore as an example to introduce the design of a unified file format and help deepen understanding of this topic.

![File format design supporting random access](../img/ch07/7.3/file_indexing.png)
:width:`800px`
:label:`file_random_access`

#### Introduction to MindRecord

MindRecord is the unified data format introduced by MindSpore, with the goal of normalizing user datasets and optimizing the training data loading process. This file format has the following characteristics:

-   Enables unified storage and access of diverse user data, making training data loading more convenient.

-   Aggregated data storage for efficient reading, while being easy to manage and transfer.

-   Efficient data encoding and decoding operations, transparent and imperceptible to users.

-   Flexible control over partition sizes, facilitating distributed training.

Similar to the Unirecord design described earlier, a MindRecord file also consists of data files and index files. The data file contains a file header, scalar data pages, and block data pages for storing users' normalized training data. The index file contains index information generated based on scalar data (such as image labels, image filenames, etc.) for convenient retrieval and statistical analysis of dataset information. To ensure random access performance for a single MindRecord file, MindSpore recommends that each MindRecord file be smaller than 20 GB. If a dataset exceeds 20 GB, users can specify the corresponding parameters during MindRecord dataset generation to shard the original dataset into multiple MindRecord files.

![MindRecord file format composition](../img/ch07/7.3/MindRecord_format.png)

:width:`800px`
:label:`mindrecord_format`

The detailed information about the key components of the data file portion in a MindRecord file is as follows:

-   **File Header**
    The file header is primarily used to store the file header size, scalar data page size, block data page size, Schema information, index fields, statistical information, file partition information, and the correspondence between scalar data and block data. It serves as the metadata of the MindRecord file.

-   **Scalar Data Pages**
    Scalar data pages are primarily used to store integer, string, and floating-point data, such as image labels, image filenames, image dimensions, and other information that is suitable for scalar storage.

-   **Block Data Pages**
    Block data pages are primarily used to store binary strings, NumPy arrays, and similar data, such as binary image files themselves and dictionaries converted from text.

During training, MindRecord's reader can quickly locate and find the position of data based on index files, and read and decode the data. Additionally, MindRecord possesses certain retrieval capabilities, allowing users to filter and obtain data samples that meet their expectations by specifying query conditions.

For distributed training scenarios, MindRecord loads metadata based on the Header in data files and index files to obtain the IDs of all samples and their offset information within data files. It then performs data partitioning based on user-input num_shards (number of training nodes) and shard_id (current node ID), obtaining 1/num_shards of the data for the current node. In other words, during distributed training, multiple nodes each read only 1/num_shards of the dataset, and the effect of training on the entire dataset is achieved through AllReduce on the computation side. Furthermore, if users enable the shuffle operation, the shuffle seed is kept consistent across all nodes within each epoch, ensuring that the ID shuffle results for all samples are consistent, which in turn ensures correct data partitioning.

![MindRecord Partition strategy](../img/ch07/7.3/partition.png)

:width:`800px`
:label:`mindrecord_partition`

### Efficiency of Data Computation

After addressing the data loading performance issue, let us continue to study how to improve data computation performance (i.e., maximizing the data processing rate P mentioned earlier). We will use the data preprocessing pipeline mentioned above as an example to study how to design the data module's scheduling and execution of user computation graphs to achieve optimal performance.

![Diagram of serialized sequential execution of data preprocessing](../img/ch07/7.3/single_pipeline.png)

:width:`800px`
:label:`serialized_data_process`

Since deep learning chips such as GPUs and Huawei Ascend do not possess general-purpose data processing capabilities,
we currently still rely primarily on CPUs to complete preprocessing computation. Mainstream AI servers are equipped with multiple multi-core CPUs, and the data module needs to design reasonable parallel architectures to fully leverage multi-core computing power, thereby improving data preprocessing performance and minimizing accelerator stalls caused by waiting for data. In this section, we will introduce two common parallel architectures: pipeline-level parallelism and operator-level parallelism. Pipeline parallelism has a clear structure, is easy to understand and implement, and is primarily adopted by machine learning systems like PyTorch that implement data modules in Python. Influenced by the scheduling and execution architecture designs of classic data-parallel systems, other systems such as Google's TensorFlow and Huawei's MindSpore primarily adopt operator-level parallelism for fine-grained CPU resource allocation to fully utilize multi-core computing power. However, fine-grained allocation means we need to set reasonable parallelism parameters for all operators involved in the data processing pipeline, which poses a significant challenge for users. Consequently, frameworks like MindSpore also provide automatic tuning of key parameters in the data flow graph. Through dynamic analysis at runtime, the system automatically searches for optimal operator parallelism parameters, greatly reducing the user's programming burden. Let us now discuss each approach in detail.

#### Pipeline Parallelism

The first common parallelism approach is pipeline-level parallelism, where the user's constructed computation pipeline is executed sequentially within a single thread/process, while multiple threads/processes are launched to execute multiple pipelines in parallel. If users need to process a total of N data samples, then with pipeline parallelism degree M, each process/thread only needs to process (N/M) samples. Pipeline parallelism has a simple architecture and is easy to implement. Within the entire parallel architecture, each executing process/thread only needs to communicate across processes/threads at the beginning and end of data execution. The data module distributes pending data tasks to each pipeline process/thread and finally aggregates the results to send to the chip for model computation. From the user's perspective, usage is also relatively convenient, requiring only the specification of the key parallelism degree parameter. Let us use PyTorch as an example for detailed elaboration.

![Diagram of pipeline-level parallel execution](../img/ch07/7.3/pipeline_parallisim.png)
:width:`800px`
:label:`pipeline_parallisim`

In PyTorch, users only need to implement a Dataset Python class to write the data processing logic. The Dataloader launches the corresponding number of Python processes based on the user-specified parallelism parameter num_workers to invoke the user-defined Dataset class for data preprocessing. The Dataloader has two types of process roles: worker processes and the main process, along with two types of inter-process communication queues: index_queue and worker_result_queue. During training, the main process sends the list of pending data tasks to each worker process through index_queue. Each worker process executes the data preprocessing logic of the user-written Dataset class and returns the processed results to the main process through worker_result_queue.

![PyTorch Dataloader parallel execution architecture](../img/ch07/7.3/pytorch_dataloader.png)
:width:`800px`
:label:`pytorch_dataloader`

Next, we present a code snippet of using PyTorch's Dataloader for parallel data preprocessing. We can see that we only need to implement the Dataset class to describe the data preprocessing logic and specify num_workers to achieve pipeline-level parallel data preprocessing.


```python
# Describe the data preprocessing workflow
class TensorDataset:
    def __init__(self, inps):
        sef.inps = inps

    def __getitem__(self, idx):
        data = self.inps[idx]
        data = data + 1
        return data

    def __len__(self):
        return self.inps.shape[0]

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps)

# Set parallelism degree to 3
loader = DataLoader(dataset, batch_size=2, num_workers=3)

for batch_idx, sample in enumerate(loader):
    print(sample)
```

Finally, it should be noted that PyTorch Dataloader's execution involves extensive inter-process communication. Although PyTorch has implemented shared memory-based inter-process communication for Tensor-type data to accelerate this step, when the communication data volume is large, cross-process communication can still significantly impact end-to-end data preprocessing throughput performance. Of course, this is not an architectural issue with pipeline parallelism itself, but rather a consequence of CPython's Global Interpreter Lock (GIL), which forces pipeline parallelism at the Python level to use process parallelism rather than thread parallelism. To address this issue, the PyTorch team is currently attempting to remove the GIL from CPython to achieve thread-based pipeline parallelism for improved communication efficiency :cite:`rmpygil`. Interested readers can explore this topic further.

#### Operator Parallelism

In pipeline parallelism, computing resources (CPU cores) are allocated at the pipeline granularity. In contrast, operator parallelism allocates resources at the operator granularity, pursuing a more fine-grained resource allocation approach. We aim to assign higher parallelism to operators with greater computation costs and lower parallelism to operators with lesser computation costs, achieving more efficient and reasonable CPU resource utilization. The idea of operator parallelism is in the same spirit as classic data-parallel computing system parallelism. Taking classic MapReduce execution as an example, we can see that this can also be considered a form of operator parallelism (map operators and reduce operators), where the parallelism degree of map operators and reduce operators is determined by the computation cost of each operator phase.

![Classic MapReduce parallel execution architecture](../img/ch07/7.3/map_reduce.png)
:width:`800px`
:label:`mapreduce`

In the figure below, we present the operator parallelism architecture diagram for the data preprocessing pipeline introduced at the beginning of this section. Based on the computation cost of each operator, we set the image decoding operator parallelism to 3, image resizing parallelism to 2, image random rotation operator parallelism to 4, image normalization operator parallelism to 3, and image channel transposition operator parallelism to 1. We aim to achieve efficient and full utilization of computing resources by precisely allocating resources to operators with different computation costs. In specific implementations, operator parallelism generally uses thread-level parallelism, with all operators communicating through shared memory using inter-thread queues and similar methods.

![Operator parallel execution architecture](../img/ch07/7.3/operator_parallisim.png)
:width:`800px`
:label:`operator_parallisim`

Among existing machine learning system data modules, tf.data and MindData both adopt the operator parallelism approach. Due to more efficient resource utilization and high-performance data flow scheduling implemented in C++, operator parallelism approaches often demonstrate better performance. Performance evaluations of tf.data show that it has nearly twice the performance advantage compared to PyTorch's Dataloader :cite:`murray2021tf`.
Next, we use a MindSpore-based implementation of the data preprocessing pipeline described at the beginning of this section to demonstrate how to set the parallelism degree for each operator in an operator-parallel data pipeline.

```python
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.vision.c_transforms as vision

# Load data
dataset_dir = "path/to/imagefolder_directory"
dataset = ds.ImageFolderDatasetV2(dataset_dir, num_parallel_workers=8)
transforms_list = [vision.Decode(),
                   vision.Resize((256, 256)),
                   vision.RandomRotation((0, 15)),
                   vision.Normalize((100,  115.0, 121.0), (71.0, 68.0, 70.0)),
                   vision.HWC2CHW()]
onehot_op = c_transforms.OneHot(num_classes)
# Decoding operator parallelism degree: 3
dataset = dataset.map(input_columns="image", operations=vision.Decode(), num_parallel_workers=3)
# Resizing operator parallelism degree: 2
dataset = dataset.map(input_columns="image", operations=vision.Resize((256, 256)), num_parallel_workers=2)
# Random rotation operator parallelism degree: 4
dataset = dataset.map(input_columns="image", operations=vision.RandomRotation((0, 15)), num_parallel_workers=4)
# Normalization operator parallelism degree: 3
dataset = dataset.map(input_columns="image", operations=vision.Normalize((100,  115.0, 121.0), (71.0, 68.0, 70.0)), num_parallel_workers=3)
# Channel transposition operator parallelism degree: 1
dataset = dataset.map(input_columns="image", operations=vision.HWC2CHW(), num_parallel_workers=1)
dataset = dataset.map(input_columns="label", operations=onehot_op)
```

We observe that while operator parallelism has higher performance potential, it requires us to set reasonable parallelism parameters for each operator. This not only places high demands on users but also increases the risk of performance degradation due to unreasonable parameter settings. To make operator parallelism easier for users, both tf.data and MindData have added dynamic tuning of key pipeline parameters, computing reasonable parameters based on runtime performance monitoring of the pipeline execution to achieve optimal data preprocessing throughput as much as possible :cite:`murray2021tf`.

#### Data Processing Computation Graph Optimization

In the preceding text, we focused on efficiently executing the user's constructed data preprocessing computation graph through parallel architectures. However, we can consider the following question: Is the computation graph given by the user an efficient one?
If not, can we optimize and rewrite the user's data computation graph under the premise of equivalent transformation to obtain a computation graph with expected better execution performance? Indeed, this shares the same philosophy as the model computation graph compilation optimization we studied in previous chapters --- that is, achieving better execution performance by analyzing and transforming the computation graph IR to obtain a more optimal IR representation. Common data graph optimization strategies include operator fusion and map operation vectorization. Operator fusion merges operator combinations such as map+map, map+batch, map+filter, and filter+filter into equivalent composite operators, combining computations that originally required execution in two thread groups into composite computations executed in a single thread group. This reduces inter-thread synchronization and communication overhead, achieving better performance. Map operation vectorization transforms the common dataset.map(f).batch(b) operation combination into dataset.batch(b).map(parallel_for(f)), leveraging modern CPUs' parallelism-friendly SIMD instruction sets to accelerate data preprocessing.