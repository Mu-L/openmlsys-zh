
## Background

Throughout human history, technological progress, production relations, and the development of ethical regulations have evolved dynamically. When a new technology achieves a breakthrough in the laboratory, the resulting changes in value creation sequentially impact commodity forms, production relations, and other aspects. At the same time, once the value gains brought by new technology are recognized, the organizational forms of business logic, in their spontaneous adjustment process, also place demands on the path, content, and even pace of technological development, and adapt new ethical regulations when these demands are met. Through such interactions, technological systems and social systems resonate and co-evolve---this is what constitutes a technological revolution.

Over the past decade, driven by the cost-performance ratio of computational power and data scale surpassing critical thresholds, connectionist model architectures represented by deep neural networks and statistical learning paradigms (hereinafter referred to as deep learning) have achieved breakthrough advances in feature representation capabilities, greatly advancing the development of artificial intelligence and achieving remarkable results in many scenarios. For example, face recognition accuracy has reached over 97%, and Google's intelligent voice assistant achieved a 92.9% correct response rate in 2019 tests. In these typical scenarios, deep learning's intelligent performance has surpassed that of ordinary humans (and even experts), reaching a tipping point for technology replacement. In recent years, in domains where business logic is technology-friendly or where ethical regulations are temporarily sparse---such as security, real-time scheduling, process optimization, competitive gaming, and information feed distribution---artificial intelligence and deep learning have achieved rapid technical and commercial breakthroughs.

Having tasted success, no domain wants to miss out on the benefits of technological progress. However, when the commercial application of deep learning enters domains that are technology-sensitive and closely related to human survival or safety---such as autonomous driving, finance, healthcare, and judicial high-risk application scenarios---the existing business logic encounters resistance during technology replacement, leading to slowdowns or even failures in commercialization. The root cause is that the business logic and underlying ethical regulations of these scenarios center on stable, traceable accountability and responsibility distribution; yet the models produced by deep learning are black boxes from which we cannot extract any information about model behavior from the model's structure or weights, rendering the accountability and distribution mechanisms in these scenarios inoperative and causing technical and structural difficulties for AI in business applications.

Here are two specific examples: Example 1, in the financial risk control scenario, a deep learning model identifies a small subset of users with suspected fraud, but the business department does not dare to directly act on these results. Because people cannot understand how the results were obtained, they cannot determine whether the results are accurate. Moreover, the results lack clear evidence, and if acted upon, cannot be justified to regulatory agencies.
Example 2, in the medical field, a deep learning model determines that a patient has tuberculosis based on the patient's test data, but the doctor does not know how the diagnosis was reached and does not dare to directly adopt it, instead relying on their own experience, carefully reviewing the relevant test data, and then making their own judgment. These two examples demonstrate that black-box models seriously hinder the application and promotion of models in real-world scenarios.

Moreover, model interpretability has attracted national-level attention, with relevant institutions issuing related policies and regulations.

-   In July 2017, the State Council issued the "New Generation Artificial Intelligence Development Plan," which for the first time encompassed explainable AI.

-   In March 2021, the People's Bank of China released the financial industry standard "Evaluation Specification for Financial Applications of Artificial Intelligence Algorithms," which set explicit requirements for the interpretability of AI models in the financial industry.

-   In August 2021, the Cyberspace Administration of China issued the "Provisions on the Management of Algorithmic Recommendations for Internet Information Services," proposing requirements for the interpretability of algorithmic recommendations in the internet industry.

-   In September 2021, the Ministry of Science and Technology released the "Ethical Norms for New Generation Artificial Intelligence."

Therefore, from both the commercial promotion and regulatory perspectives, we need to open up the black box model and provide explanations for models. Explainable AI is precisely the technology that addresses this class of problems.

## Definition of Explainable AI

According to DARPA (Defense Advanced Research Projects Agency), as shown in :numref:`xai_concept`,
the concept of explainable AI is: unlike existing AI systems, explainable AI systems can address the problems users face with black-box models, enabling users to know not only what, but also why.


![Concept of Explainable AI (Image source: Broad Agency Announcement Explainable Artificial Intelligence (XAI) DARPA-BAA-16-53)](../img/ch11/xai_concept.png)
:width:`800px`
:label:`xai_concept`

However, neither academia nor industry has a unified definition of explainable AI (eXplainable AI, XAI). Here we list three typical definitions for discussion:

-   Interpretability is the desire to directly understand the working mechanism of a model, breaking open the black box of artificial intelligence.

-   Explainable AI provides human-readable and understandable explanations for decisions made by AI algorithms.

-   Explainable AI is a set of methods that ensures humans can easily understand and trust the decisions made by AI agents.


Based on our practical experience and understanding, we define explainable AI as: a collection of techniques oriented toward machine learning (primarily deep neural networks), including visualization, data mining, logical reasoning, knowledge graphs, etc. The purpose is to use this collection of techniques to make deep neural networks exhibit a certain degree of understandability, so as to satisfy the information needs (such as causal or background information) of relevant users regarding models and application services, thereby establishing cognitive-level trust in AI services among users.

## Overview of Explainable AI Algorithms

With the emergence of the concept of explainable AI, XAI has received increasing attention from both academia and industry. The figure below shows the trend of explainable AI keywords in top academic conferences in the field of artificial intelligence. To provide readers with a holistic understanding of existing explainable AI algorithms, we summarize and categorize the types of XAI algorithms with reference to :cite:`2020tkde_li`, as shown in :numref:`XAI_methods`.

![Explainable AI (XAI) algorithm branches](../img/ch11/XAI_methods.PNG)
:width:`800px`
:label:`XAI_methods`

There are diverse methods for explaining models. Here, based on whether the explanation process introduces external knowledge beyond the dataset, we divide them into data-driven explanation methods and knowledge-aware explanation methods.

**Data-Driven Explanations**

Data-driven explanations refer to methods that generate explanations purely from the data itself, without requiring external information such as prior knowledge. To provide explanations, data-driven methods typically start by selecting a dataset (with global or local distribution). Then, the selected dataset or its variants are fed into the black-box model (in some cases, selecting a dataset is not necessary; for example, the maximum activation method proposed by :cite:`erhan2009visualizing`), and explanations are generated through certain analysis of the corresponding predictions from the black-box model (e.g., computing derivatives of predictions w.r.t. input features). Based on the scope of interpretability, these methods can be further divided into global methods or local methods---that is, whether they explain the global model behavior across all data points or the behavior of a subset of predictions. In particular, instance-based methods provide a special type of explanation---they directly return data instances as explanations. Although from the perspective of explanation scope, instance-based methods can also fit into global methods (representative samples) or local methods (counterfactuals), we list them separately to emphasize their distinctive way of providing explanations.

Global methods aim to provide an understanding of the model logic and complete reasoning for all predictions, based on a holistic view of its features, learned components, and structure. Several directions can be explored for global interpretability. For ease of understanding, we divide them into the following three subcategories:
(i)
Model extraction---extracting an interpretable model from the original black-box model, for example, distilling the original black-box model into an interpretable decision tree through model distillation :cite:`frosst2017distilling` :cite:`zhang2019interpreting`, thereby using the rules in the decision tree to explain the original model;
(ii)
Feature-based methods---estimating feature importance or relevance, as shown in :numref:`xai_global_feature_importance`.
This type of explanation can provide explanations such as "credit overdue records are the most important feature relied upon by the model," thereby helping to determine whether the model has bias. A typical global feature explanation method is SHAP (which can only output global explanations for tree models) :cite:`lundberg2017unified`.
(iii) Transparent model design---modifying or redesigning black-box models to improve their interpretability. This class of methods has also gradually become a research hotspot, with recent related work including ProtoPNet :cite:`chen2019looks`, Interpretable CNN :cite:`zhang2018interpretable`, ProtoTree :cite:`nauta2021neural`, etc.

![Global feature importance explanation](../img/ch11/xai_global_feature_importance.png)
:width:`800px`
:label:`xai_global_feature_importance`


Global explanations can provide an overall understanding of the black-box model. However, due to the high complexity of black-box models, in practice it is often difficult to obtain simple transparent models with behavior similar to the original model through model extraction/design, and it is often difficult to abstract unified feature importance across the entire dataset. Furthermore, global explanations also lack local fidelity when generating explanations for individual observations, as globally important features may not accurately explain decisions for individual samples. Therefore, local methods have become an important research direction in recent years. Local methods attempt to verify the reasonableness of model behavior for individual instances or a set of instances. When focusing only on local behavior, complex models can become simple, so even simple functions can provide highly credible explanations for local regions. Based on the process of obtaining explanations, local methods can be divided into two categories: local approximation and propagation-based methods.

Local approximation generates understandable sub-models by simulating the behavior of the black-box model in the neighborhood of a sample. Compared to model extraction in global methods, local approximation only needs to focus on the neighborhood of the sample, making it easier to obtain sub-models that accurately describe local behavior. As shown in :numref:`xai_lime`, by generating $m$ data points $(x_i^\prime, f(x_i^\prime)), for\  i=1,2, ...m$ (where $f$ is the black-box model decision function) near the data point of interest $x$, and linearly fitting these data points, we can obtain a linear model $g=\sum_i^k w_ix^i$, where $k$ represents the feature dimensionality of the data. The weights $w_i$ in the linear model can then be used to represent the importance of the $i$-th feature of data $x$ for model $f$.

![Example of local approximation method](../img/ch11/xai_lime.png)
:width:`800px`
:label:`xai_lime`

Propagation-based methods typically propagate certain information to directly locate relevant features. These methods include backpropagation-based methods and forward propagation-based methods. Backpropagation-based methods attribute the output contributions to input features through gradient backpropagation. As shown in :numref:`xai_gradient_based`, through gradient backpropagation, the gradient of the model output with respect to the input $\frac{d(f(x))}{dx}$ is computed as the model explanation. Common gradient propagation-based methods include the basic Gradient method, GuidedBackprop :cite:`zeiler2014visualizing`, GradCAM :cite:`selvaraju2017grad`, etc.
Forward propagation-based methods quantify the correlation between outputs and features by perturbing features and observing the differences in forward inference outputs. Common methods in this category include RISE :cite:`petsiuk2018rise`, ScoreCAM :cite:`wang2020score`, etc.


![Example of gradient-based method](../img/ch11/xai_gradient_based.PNG)
:width:`800px`
:label:`xai_gradient_based`

**Knowledge-Aware Explanations**

Data-driven explanation methods can provide comprehensive explanations from datasets or the relationships between inputs and outputs. Building on this, external knowledge can also be leveraged to enrich explanations and make them more human-friendly. Laypersons without machine learning background knowledge may find it difficult to directly understand feature importance and the connections between features and targets. With external domain knowledge, we can not only generate explanations indicating feature importance, but also describe why certain features are more important than others. Therefore, knowledge-aware explainable AI methods have attracted increasing attention in recent years. Compared to raw datasets collected from multiple scenarios, knowledge is typically regarded as entities or relationships derived from human life experience or rigorous theoretical reasoning. Generally, knowledge can take many forms. It can reside in people's minds, or be recorded in natural language, audio, or rules with strict logic. To systematically review these methods, we categorize them based on knowledge sources into two types: general knowledge methods and knowledge base (KB) methods. The former uses unstructured data as a knowledge source to construct explanations, while the latter uses structured knowledge bases as the foundation for building explanations.

A relatively straightforward approach to providing knowledge is through human involvement. In fact, with the explosive growth of AI research and applications, the critical role of humans in AI systems has gradually become apparent. Such systems are called human-centered AI systems. :cite:`riedl2019human` argue that human-centered AI can not only enable AI systems to better understand humans from a sociocultural perspective, but also enable AI systems to help humans understand themselves. To achieve these goals, AI needs to satisfy several properties including interpretability and transparency.


Specifically, humans can play a role in AI systems by providing a considerable number of human-defined concepts. :cite:`kim2018interpretability` uses Concept Activation Vectors (CAV) to test the importance of concepts in classification tasks (TCAV). A CAV is a vector perpendicular to the decision boundary between the activation and non-activation of a target concept of interest. This vector can be obtained as follows: input positive and negative samples of the target concept, perform linear regression to get the decision boundary, and thereby obtain the CAV. Taking the "stripes" concept for "zebra" as an example, the user first collects data samples containing "stripes" and data samples not containing "stripes," feeds them into the network, obtains the activation values of intermediate layers, fits these based on positive and negative sample labels ($1$ for containing the concept, $0$ for not containing the concept) to the intermediate layer activation values, obtains the decision boundary, and the CAV is the perpendicular vector to this decision boundary.


As shown in :numref:`xai_tcav`, to compute the TCAV score, the "concept sensitivity" representing the importance of a concept at layer $l$ for class $k$ prediction can first be computed as the directional derivative $S_{C,k,l}(\mathbf{x})$:
$$\begin{split}
S_{C,k,l}(\mathbf{x}) =  &\lim_{\epsilon\rightarrow 0}\frac{h_{l,k}(f_{l}(\mathbf{x})+\epsilon \mathbf{v}^{l}_{C})-h_{l,k}(f_{l}(\mathbf{x}))}{\epsilon} \\ = &\nabla h_{l,k}(f_{l}(\mathbf{x})) \cdot \mathbf{v}^{l}_{C}
\end{split}
\label{eq:TCAV_score}$$
where $f_{l}(\mathbf{x})$ is the activation at layer $l$, $h_{l,k}(\cdot)$ is the logit for class $k$, $\nabla h_{l,k}(\cdot)$ is the gradient of $h_{l,k}$
w.r.t. the activations at layer $l$. $\mathbf{v}^{l}_{C}$ is the CAV for concept $C$ that the user aims to explore. Positive (or negative) sensitivity indicates that concept $C$ has a positive (or negative) influence on the activation of the input.

Based on $S_{C,k,l}$,
TCAV can then be obtained by computing the ratio of samples of class $k$ with positive $S_{C,k,l}$'s:

$$\textbf{TCAV}_{Q_{C,k,l}}=\frac{\vert \{\mathbf{x}\in X_{k}:S_{C,k,l}(\mathbf{x})>0\}\vert}{\vert X_{k}\vert}
\label{eq:TCAV}$$
Combined with the $t$-distribution hypothesis method, if $\textbf{TCAV}_{Q_{C,k,l}}$ is greater than 0.5, it indicates that concept $C$ has a significant influence on class $k$.

![TCAV pipeline (Image source: :cite:`2020tkde_li`)](../img/ch11/xai_tcav.png)
:width:`800px`
:label:`xai_tcav`

Human knowledge can be subjective, while KB can be objective. In current research, KB is usually modeled as a Knowledge Graph (KG). The following uses the explainable recommendation model TB-Net, supported by MindSpore, as an example to explain how to build an explainable model using knowledge graphs. Knowledge graphs can capture rich semantic relationships between entities. One of TB-Net's objectives is to identify which pair of entities (i.e., item-item) has the most significant influence on the user, and through which relationships and key nodes they are connected. Unlike existing KG embedding-based methods (RippleNet uses KG completion methods to predict paths between users and items), TB-Net extracts real paths to achieve high accuracy and superior interpretability of recommendation results.

![TB-Net network training framework](../img/ch11/tb_net.png)
:width:`800px`
:label:`tb_net`

The framework of TB-Net is shown in :numref:`tb_net`: where $i_c$ represents the candidate item to be recommended, $h_n$ represents items that the user has interacted with in their history, $r$ and $e$ represent relations and entities in the knowledge graph, and their vectorized representations are concatenated to form relation matrices and entity matrices. First, TB-Net constructs a subgraph for user $u$ by connecting $i_c$ and $h_n$ through shared attribute values. Each pair of $i_c$ and $h_n$ is connected by a path composed of relations and entities. Then, TB-Net's bidirectional path propagation method propagates the computation of item, entity, and relation vectors from the left and right sides of the path to the middle node, computing the probability that the two directional flows converge at the same intermediate entity. This probability is used to represent the user's preference for the intermediate entity and serves as the basis for explanations. Finally, TB-Net identifies key paths (i.e., key entities and relations) in the subgraph, outputting recommendation results and explanations with semantic-level detail.

Taking game recommendation as a scenario, randomly recommending a new game to a user, as shown in :numref:`xai_kg_recommendation`, where Half-Life, DOTA 2, Team Fortress 2, etc. are game titles. In the relation attributes, game.year represents the game release year, game.genres represents game genre, game.developer represents the game developer, and game.categories represents game categories. In the attribute nodes, MOBA stands for Multiplayer Online Battle Arena, Valve is the Valve Corporation, Action stands for action genre, Multi-player stands for multiplayer mode, Valve Anti-Cheat enabled represents the Valve Anti-Cheat system, Free means free-to-play, and Cross-Platform means cross-platform support. The games on the right are games the user has played according to their history. The correctly recommended game in the test data is "Team Fortress 2."

![Steam game recommendation explainability example (Games played by user: Half-Life, DOTA 2. Correctly recommended game: "Team Fortress 2." Nodes with attribute information such as game.genres: Action, free-to-play; game.developer: Valve; game.categories:
Multiplayer, MOBA.)](../img/ch11/xai_kg_recommendation.png)
:width:`800px`
:label:`xai_kg_recommendation`

In :numref:`xai_kg_recommendation`, there are two highlighted relevance probabilities (38.6%, 21.1%), which are the probabilities of key paths being activated during the recommendation process as computed by the model. The red arrows highlight the key path from "Team Fortress 2" to the historical item "Half-Life." This shows that TB-Net can recommend items to users through various relational connections and identify key paths as explanations. Therefore, the explanation for recommending "Team Fortress 2" to the user can be translated into a fixed narrative: "Team Fortress 2 is an action, multiplayer online, shooting video game developed by game company Valve. It is highly correlated with the game Half-Life that the user has played before."

## Explainable AI Systems and Practice

As the demand for explainability grows rapidly across various domains, an increasing number of enterprises are integrating explainable AI toolkits to provide users with fast and convenient explainability solutions. The mainstream toolkits currently available in the industry include:
- TensorFlow team's What-if Tool, which allows users to explore learning models without writing any code, enabling non-developers to participate in model tuning.
- IBM's AIX360, which provides multiple explanation and measurement methods to evaluate model interpretability and trustworthiness across different dimensions.
- Facebook's Torch team's Captum, which offers multiple mainstream explanation methods for image and text scenarios.
- Microsoft's InterpretML, which allows users to train different white-box models and explain black-box models.
- SeldonIO's Alibi, which focuses on inspecting model internals and decision explanations, providing implementations of various white-box, black-box, single-sample, and global explanation methods.
- Huawei MindSpore's XAI tool, which provides data tools, explanation methods, white-box models, and measurement methods, offering users explanations at different levels (local, global, semantic-level, etc.).

This section uses the MindSpore XAI tool as an example to explain how to use explainable AI tools in practice to provide explanations for image classification models and tabular data classification models, thereby helping users understand models for further debugging and optimization.
The architecture of the MindSpore XAI tool is shown below. It is an explainability tool built on the MindSpore deep learning framework and can be deployed on Ascend and GPU devices.
![MindSpore XAI architecture diagram](../img/ch11/mindspore_xai.png)
:width:`800px`
:label:`mindspore_xai`

To use MindSpore Explainable AI, readers first need to install the MindSpore XAI package via pip (supporting MindSpore 1.7 or above, GPU and Ascend processors, recommended to use with JupyterLab):

```bash
pip install mindspore-xai
```

In the MindSpore XAI [official tutorial](https://www.mindspore.cn/xai/docs/zh-CN/r1.8/index.html), detailed instructions on how to install and use the provided explanation methods are available for readers to consult.

### MindSpore XAI Tool for Image Classification Explanation

Below is a code demonstration example combining the saliency map visualization method GradCAM, which is supported in MindSpore XAI version 1.8. Readers can refer to the [official tutorial](https://www.mindspore.cn/xai/docs/zh-CN/1.8/using_cv_explainers.html) to obtain the demo dataset, model, and complete script code.

```python

from mindspore_xai.explainer import GradCAM

# Typically specify the last convolutional layer
grad_cam = GradCAM(net, layer="layer4")

# 3 is the ID for the 'boat' class
saliency = grad_cam(boat_image, targets=3)
```

If the input is an image tensor of dimension $1*3*224*224$, the returned saliency is a saliency map tensor of dimension $1*1*224*224$. Below we present several examples demonstrating how to use explainable AI capabilities to better understand the prediction results of image classification models, identify the key feature regions used as the basis for classification predictions, and thereby judge the reasonableness and correctness of the classification results to accelerate model optimization.


![Example where the prediction result is correct and the key features relied upon are reasonable](../img/ch11/correct_correct.png)
:width:`400px`
:label:`correct_correct`

In the figure above, the predicted label is "bicycle," and the explanation result shows that the key features relied upon are on the wheels, indicating that this classification judgment basis is reasonable and the model can be preliminarily deemed trustworthy.

![Example where the prediction result is correct but the key features relied upon are unreasonable](../img/ch11/correct_wrong.png)
:width:`400px`
:label:`correct_wrong`

In the figure above, one of the predicted labels is "person," which is correct. However, in the explanation, the highlighted region is on the horse's head, so the key feature basis is likely incorrect, and the reliability of this model needs further verification.

![Example where the prediction result is incorrect and the key features relied upon are unreasonable](../img/ch11/wrong_wrong.png)
:width:`400px`
:label:`wrong_wrong`

In the figure above, the predicted label is "boat," but there is no boat in the original image. Through the explanation result on the right side of the figure, we can see that the model used the water surface as the key basis for classification to arrive at the prediction "boat"---this basis is incorrect. By analyzing the subset of the training dataset labeled "boat," it was found that the vast majority of images labeled "boat" contain water surfaces, which likely caused the model to mistakenly learn water surfaces as a key feature for the "boat" class during training. Based on this finding, proportionally supplementing images with boats but without water surfaces can significantly reduce the probability of the model misjudging key features during learning.

### MindSpore XAI Tool for Tabular Classification Explanation
MindSpore XAI version 1.8 supports three commonly used tabular data model explanation methods in the industry: LIMETabular, SHAPKernel, and SHAPGradient.

Using LIMETabular as an example, it provides a locally interpretable model to explain individual samples for a complex, hard-to-explain model:
```python
from mindspore_xai.explainer import LIMETabular

# Convert features to feature statistics
feature_stats = LIMETabular.to_feat_stats(data, feature_names=feature_names)

# Initialize the explainer
lime = LIMETabular(net, feature_stats, feature_names=feature_names, class_names=class_names)

# Explain
lime_outputs = lime(inputs, targets, show=True)
```

The explainer displays the decision boundary for classifying the sample as setosa. The returned lime_outputs is a structured data representing the decision boundary.
Visualizing the explanation yields
![LIME explanation result](../img/ch11/tabular.png)
:width:`400px`
:label:`tabular_lime`
The above explanation shows that for the setosa classification decision, the most important feature is petal length.

### MindSpore XAI Tool: White-Box Models

In addition to post-hoc explanation methods for black-box models, the XAI tool also provides industry-leading white-box models, enabling users to train on these white-box models so that during inference the model can simultaneously output both inference results and explanations. Taking TB-Net as an example (refer to :numref:`tb_net` and its [official tutorial](https://e.gitee.com/mind_spore/repos/mindspore/xai/tree/master/models/whitebox/tbnet) for usage), this method has been deployed commercially, providing millions of customers with semantic-level explainable financial product recommendation services. TB-Net leverages knowledge graphs to model the attributes of financial products and customers' historical data. In the graph, financial products with common attribute values are connected. The candidate product and the customer's historically purchased or browsed products are connected through common attribute values into paths, forming the customer's subgraph. Then, TB-Net performs bidirectional propagation computation on the paths in the graph to identify key products and key paths as the basis for recommendations and explanations.


An example of explainable recommendation is as follows: in the historical data, the customer has recently purchased or browsed financial products A, B, N, etc. Through TB-Net's bidirectional path propagation computation, it is found that the path (Product P, moderate-to-high annualized return, Product A) and the path (Product P, moderate risk level, Product N) have high weights, making them key paths. At this point, TB-Net outputs the following explanation: "Financial product P is recommended to this customer because its moderate-to-high annualized return and moderate risk level are consistent with financial products A and B that the customer has recently purchased or browsed."

![TB-Net application in financial wealth management scenario](../img/ch11/tbnet_finance.png)
:width:`800px`
:label:`tbnet_finance`

In addition to the explanation methods introduced above, MindSpore XAI also provides a series of measurement methods for evaluating the quality of different explanation methods, and will continue to add white-box models with built-in explanations. Users can directly adopt mature model architectures to quickly build their own explainable AI systems.


## Future of Explainable AI

To further advance research in explainable AI, we summarize several noteworthy research directions here.

First, knowledge-aware XAI still has significant room for research expansion. However, there are still many open questions regarding how to effectively leverage external knowledge. One issue is how to acquire or retrieve useful knowledge from such a vast knowledge space. For example, Wikipedia contains knowledge related to various fields, but if the goal is to solve a medical image classification problem, most Wikipedia entries are irrelevant or noisy, making it difficult to accurately find appropriate knowledge to incorporate into the XAI system.

Furthermore, the deployment of XAI systems also urgently needs a more standardized and unified evaluation framework. To build such a standardized and unified evaluation framework, we may need to simultaneously leverage different metrics that complement each other. Different metrics may be applicable to different tasks and users. A unified evaluation framework should have corresponding flexibility.

Finally, we believe that interdisciplinary collaboration will be beneficial. The development of XAI requires not only computer scientists to develop advanced algorithms, but also physicists, biologists, and cognitive scientists to unravel the mysteries of human cognition, as well as domain experts to contribute their domain knowledge.

## References

:bibliography:`../references/explainable.bib`