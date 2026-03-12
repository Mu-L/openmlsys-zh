## Order Preservation Design

Unlike conventional data-parallel computing tasks, parallel data processing in machine learning scenarios needs to maintain order preservation to ensure experimental reproducibility. In concrete implementations, we need to guarantee that the output order of data after parallel preprocessing remains the same as the input order (i.e., SeqB and SeqA in the figure below are identical). This ensures that the output order of the data module is uniquely determined by the output order of the data shuffling component, helping users compare and debug across different experiments. Different machine learning systems adopt different approaches to ensure order preservation. We use MindSpore's implementation as an example to deepen readers' understanding of this topic.

![Data order preservation --- ensuring SeqB is identical to SeqA](../img/ch07/7.4/data_ordering.png)
:width:`800px`
:label:`data_order_definition`

MindSpore ensures order preservation by constraining the communication behavior between operator thread groups so that the input order to the current operator's downstream operator remains the same as its own input order. Based on this recursive constraint, the output order of the last operator in the entire parallel data processing pipeline is guaranteed to be the same as the input order of the first operator. In the specific implementation, MindSpore uses a Connector as the communication component between operator thread groups. The core operations on the Connector are the Push operation by the upstream operator and the Pop operation by the downstream operator. We focus on MindSpore's constraints on these two behaviors.

The usage of Connector has the following two requirements:

-   The threads in both the data producer thread group and the data consumer thread group on either side of the Connector are numbered starting from 0.

-   The input data order of the data producers must follow a round-robin distribution across producer threads. That is, when the producer thread group size is M, producer thread 0 holds the (0 + M \* k)-th data sample, producer thread 1 holds the (1 + M \* k)-th sample, producer thread 2 holds the (2 + M \* k)-th sample, and so on (where k=0, 1, 2, 3...).

The Connector maintains the same number of queues as the number of producer threads and ensures that when data is placed into the Connector, each producer thread's data goes only into the correspondingly numbered queue. This guarantees that the distribution of data across different queues in the Connector is the same as the distribution across different producer threads (the Push function in the code snippet). Then, when the Connector's consumer thread group retrieves data from the Connector, we need to ensure that the final data distribution across different consumer threads still follows a round-robin pattern. That is, when the consumer thread group size is N, consumer thread 0 holds the (0 + N \* k)-th data sample, consumer thread 1 holds the (1 + N \* k)-th sample, consumer thread 2 holds the (2 + N \* k)-th sample, and so on (where k=0, 1, 2, 3...). To achieve this, when a consumer thread requests data from the Connector, the Connector retrieves data from the queues in a round-robin manner, subject to the constraint that the requesting consumer thread number i and the pending data index j satisfy the relationship $i=j\%N$ (where N is the number of consumer threads). If the indices do not satisfy this relationship, the request blocks and waits. Through this communication constraint mechanism, MindSpore achieves order preservation.

![MindSpore order preservation implementation](../img/ch07/7.4/mindspore_data_order.jpeg)
:width:`800px`
:label:`mindspore_data_order_implementation`