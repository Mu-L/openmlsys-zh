## Scaling Single-Machine Data Processing Performance

In the previous sections, we introduced how to accelerate data preprocessing through parallel architectures that leverage multi-core CPU computing power to meet the throughput requirements of model computation on accelerator chips for data consumption. This approach can resolve user issues in most cases. However, data consumption performance is growing rapidly year over year with the development of AI chips (i.e., model computation speed is increasing), while the data module, which primarily relies on CPU computing power, cannot benefit from hardware performance improvements due to the gradual end of Moore's Law. This makes it difficult for data production performance to achieve year-over-year breakthroughs comparable to model computation performance. Moreover, in recent years the growth rate of AI chips in AI servers has far exceeded the growth rate of CPUs, further exacerbating the contradiction between chips' data consumption demands and the data module's data production performance. Taking NVIDIA's DGX series servers as an example, the DGX-1 server is configured with 40 CPU cores and 8 GPU chips. By the next generation NVIDIA DGX-2, the number of GPU chips grew to 16, while the number of CPU cores only increased from 40 to 48. Since all GPU chips share CPU computing power during training, on average, the computing power available to each GPU chip (data consumer) decreased from 5 CPU cores/GPU with NVIDIA DGX-1 to 3 CPU cores/GPU with NVIDIA DGX-2. The CPU computing power bottleneck prevents users from achieving expected scaling performance when training with multiple cards. To address the problem of insufficient CPU computing power on a single machine, we present two currently common solutions: heterogeneous data processing acceleration based on CPU+AI chips and distributed data preprocessing scaling.

### Heterogeneous Computing-Based Data Preprocessing

Since AI chips have richer computing resources compared to CPUs, leveraging AI accelerator chips for data preprocessing when CPU computing power becomes the bottleneck is an effective approach. Although AI chips do not possess general-purpose data preprocessing capabilities, most time-consuming data preprocessing operations are Tensor-related computations, such as Fast Fourier Transform (FFT) in speech processing and denoising in image processing, enabling some operations to be offloaded to AI chips for acceleration. For example, the Dvpp module on Huawei's Ascend 310 chip is a built-in hardware decoder on the chip that offers stronger image processing performance compared to CPUs. Dvpp supports basic image processing operations such as JPEG image decoding and resizing. In actual data preprocessing, users can designate certain image processing operations to be completed on the Ascend 310 chip to improve data module performance.

```python
namespace ms = mindspore;
namespace ds = mindspore::dataset;

// Initialization operations
//...

// Build data processing operators

// 1. Decode
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// 2. Resize
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// 3. Normalize
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// 4. Center crop
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));

// Build the pipeline and specify using Ascend for computation
ds::Execute preprocessor({decode, resize, center_crop, normalize}, MapTargetDevice::kAscend310, 0);

// Execute the data processing pipeline
ret = preprocessor(image, &image);
```

Compared to Dvpp, which only supports a subset of image preprocessing operations, NVIDIA's DALI :cite:`nvidia_dali` is a more general GPU-based data preprocessing acceleration framework. DALI contains the following three core concepts:

-   DataNode: Represents a collection of Tensors

-   Operator: An operator that transforms DataNodes. Both the input and output of an Operator are DataNodes. Notably, operators in DALI can be configured to one of three different execution modes: cpu, gpu, and mixed. In cpu mode, both the operator's input and output are DataNodes on the CPU. In gpu mode, both the input and output are DataNodes on the GPU. In mixed mode, the operator's input is a CPU DataNode while the output is a GPU DataNode.

-   Pipeline: A data processing pipeline constructed by users through describing the transformation process of DataNodes using Operators

In practice, users configure whether an operator's computation is performed by the CPU or GPU by setting the operator's execution mode. DALI also has the following constraint: when an operator is in mixed or gpu mode, all of its downstream operators are mandatorily required to execute in gpu mode.

![NVIDIA DALI overview](../img/ch07/7.5/dali_overview.png)

:width:`800px`
:label:`dali_overview`

Below is an example code snippet demonstrating the construction of a data processing pipeline using DALI. We read image data from files, apply mixed-mode decoding, and then process the images through rotation and resizing operators running on the GPU before returning the results to users.
Due to its demonstrated excellent performance,
DALI is widely used in high-performance inference services and multi-card training performance optimization.


```python
import nvidia.dali as dali

pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
with pipe:
    files, labels = dali.fn.readers.file(file_root = "./my_file_root")
    images = dali.fn.decoders.image(files, device = "mixed")
    images = dali.fn.rotate(images, angle = dali.fn.random.uniform(range=(-45,45)))
    images = dali.fn.resize(images, resize_x = 300, resize_y = 300)
    pipe.set_outputs(images, labels)

pipe.build()
outputs = pipe.run()
```

### Distributed Data Preprocessing

Distributed data preprocessing is another viable solution to address insufficient CPU computing power. A common approach is to leverage existing big data computing frameworks such as Spark or Dask for data preprocessing and write the results to a distributed file system. The training machines then only need to read the preprocessed result data and proceed with training.

![Distributed data preprocessing based on third-party distributed computing frameworks](../img/ch07/7.5/distribute.png)

:width:`800px`
:label:`distributed_data_preprocess_based_on_3rd_party_software`

Although this approach is widely used in the industry, it faces three problems:

-   Since data processing and model training use different frameworks, users often need to write programs in different languages across two different frameworks, increasing the user's burden.

-   Since the data processing system and the machine learning system cannot achieve zero-copy data sharing, data serialization and deserialization often become non-negligible additional overhead.

-   Since big data computing frameworks are not entirely tailored for machine learning scenarios, certain distributed preprocessing operations such as global data shuffling cannot be efficiently implemented.

To better adapt to data preprocessing in machine learning scenarios, the distributed machine learning framework Ray leverages its own task scheduling capabilities to implement simple distributed data preprocessing ---
Ray Dataset :cite:`moritz2018ray`. Since data preprocessing and training reside within the same framework, this reduces the user's programming burden while also eliminating the additional overhead of serialization/deserialization through zero-copy data sharing. Ray Dataset supports simple parallel dataset transformation operators such as map, batch, filter, as well as some basic aggregation operators like mean. Ray
Dataset also supports sorting, random shuffling, GroupBy, and other global shuffle operations. This approach is currently under research and development and has not yet been widely adopted. Interested readers can consult relevant materials for further understanding.

```python
 ray.data.read_parquet("foo.parquet") \
    .filter(lambda x: x < 0) \
    .map(lambda x: x**2) \
    .random_shuffle() \
    .write_parquet("bar.parquet")
```