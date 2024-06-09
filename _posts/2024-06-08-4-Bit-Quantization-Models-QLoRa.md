---
layout: post
title: 'Mastering QLoRa : A Deep Dive into 4-Bit Quantization and LoRa Parameter Efficient Fine-Tuning'
date : 2024-06-08
tags: [LLM, Quantization, NF4, Inference, Transformer, KV caching, Gradient Checkpointing, LoRa, QLoRa, bitsandbytes, PEFT, VRAM]
permalink: 4-Bit-Quantization-Models-QLoRa.html
categories: 
  - Jekyll
excerpt: A comprehensive step-by-step breakdown of the `bitsandbytes` 4-bit quantization with the NF4 data type. This post intends to be a one stop comprehensive guide covering everything from quantizing large language models to fine-tuning them with LoRa, along with a detailed understanding of the inference phase and decoding strategies.
---



Have you tried to implementing QLoRa before? If you are like me and you’re not satisfied with merely skimming the surface, especially with the lack of documentation of the internals of the `bitsandbytes` library (as of the writing of this post), then this post is for you. While the HuggingFace team explains a bit about FP4 in their [launch post](https://huggingface.co/blog/4bit-transformers-bitsandbytes), they do not delve deeply into NF4, despite recommending it for better performance.

This gap prompted me to explore the library thoroughly and write the most comprehensive step-by-step breakdown of the `bitsandbytes` 4-bit quantization with the  NormalFloat (NF4) data type to date. This post also intends to be a one stop comprehensive guide covering everything from quantizing large language models to fine-tuning them with LoRa, along with a detailed understanding of the inference phase and decoding strategies.

By the end of this post, you should have a deep understanding of:

1. How LLM quantization works and the theory behind LoRa.
2. How 4-bit quantization works with the NF4 data type step-by step along with an implementation from scratch of NF4 quantization that returns the exact result as the `bitsandbytes` implementation.
3. Other memory saving techniques that are useful when dealing with limited memory.
4. The implementation of QLoRa including injection, finetuning, and final merging with the base model, with a detailed code walkthrough (PEFT code abstracts a lot of things).
5. Large language model (LLM) inference process with KV caching.
6. Decoding strategies with implementation from scratch.

<br>

# **Quantization Overview**

The concept of quantization originates from the field of signal processing, where an information signal is mapped from a continuous input domain to a discrete one. In deep learning, we use quantization to reduce the precision of neural network weights and activations because models are typically over-parameterized, meaning they include a lot of redundancy. This is typically achieved by mapping their values from floating point to a discrete set of integer numbers, e.g, converting them from Float32 into INT8 representation.

<br>

> A good quantization scheme produces a smaller model size that can simulate the performance of full precision (or half precision) models with minimal loss in accuracy. This leads to a reduced memory footprint as well as a lower I/O memory bandwidth due to less data being transferred, resulting in an overall reduced inference latency. This enables you to run or deploy LLMs on resource constrained devices or even use larger, better performing models. It can be helpful in increasing serving throughput if you have an LLM that is queried by multiple users.


<br>

<i>**So how is this mapping from floating points to integer values done?**</i> This is achieved using linear scaling with a real value factor $S$ that maps the input range to the output range in the quantized space. This scaling factor $S$ is simply the ratio of the input range to the output range $S = \frac{\beta - \alpha}{\beta\_{\text{quant}} - \alpha\_{\text{quant}}}$. The result is offsetted with an integer zero-point $Z$ value that maps zero in float range to its corresponding zero value in the integer range.

<br>

There are two types of quantization symmetric and asymmetric quantization, each computing the quantization parameters $Z$ and $S$ differently. We describe below the two approaches to quantizing real values to a signed b-width bit integer representation. Symmetric quantization creates a quantized space that is symmetric around 0, where the value 0.0 in the float space is mapped exactly to 0 in the integer space. In contrast, asymmetric quantization maps 0.0 to a specific quantized zero-point $Z$. The goal of specifying this point is to ensure that 0.0 in the input data is quantized without error, this can be useful in neural network operations that contain zeros, where we want a consistent result, an example of this would be zero padding.

<br>


![Illustration detailing both symmetric and asymmetric quantization approaches to quantizing floating-point values in 32 bits to a signed integer in b-width bits  —  Created by Author.](/assets/img/pexels/qlora/quantization.png)

{:.image-caption}
*Illustration detailing both symmetric and asymmetric quantization approaches to quantizing floating-point values in 32 bits to a signed integer in b-width bits  —  Created by Author.*

<br><br>

> The quantization process in both approaches start by determining the clipping range values, an operation often referred to as calibration. The clipping range in symmetric mode $[α , −α]$ is symmetric with respect to zero, while the clipping range $[α, β]$ in asymmetric quantization is, unsurprisingly, asymmetric around zero, that is $−α ≠ β$. Next we use the clipping range values to compute the quantization parameters $S$ and $Z$. Afterward, we scale down the input data using $S$ (also offsetting with $Z$ in asymmetric mode) and round the result to the nearest integer. If the produced value falls outside the limits of the quantized domain, we use the clip operation to restrict it within the specified bounds.


<br>


The dequantization process approximates the original floating point values from the quantized integer values by simply reversing the quantization operation. This is expressed as $x = S*x_q$ in symmetric mode and $x = S(x_q-Z)$  in asymmetric mode. Dequantization can be used for different reasons, for instance, dequantizing some output that must be fed to an operation requiring greater precision input to ensure better numerical stability or simply dequantizing the output of a network.

<br>

> It’s important to note that dequantization doesn’t accuratly reconstruct the original float data, due to the quantization error that is introduced by to the rounding and clipping operations in the quantization process. Therefore, the best quantization approach is the one that minimizes this error and effectively reduces information loss associated with reducing precision.


<br>

With two quantization schemes, the question arises, <i>**when should each approach be used?**</i>

It depends mainly on the distribution of the data that we want to quantize. If the original data is normally distributed, symmetric quantization is preferred, whereas asymmetric quantization is more suitable for skewed data.

<br>

> It’s important to note that asymmetric quantization is far more complicated than symmetric quantization to implement, as the former includes the additional $Z$ term, which during matrix multiplication, introduces computational overhead leading to increased latency. This pleads the question, if this is the case, why not simply apply symmetric quantizaton on skewed data? Let’s explore this by considering an example with one of most common operations in neural network that systematically generates skewed data, the ReLU activation function.

<br>


Before delving into the case study, it’s important to point out that the representational capacity of a $k$-bit width refers to the number of distinct values that can be represented with $k$-bit width, which is $2^k$ possible unique values. For instance, 8-bit quantization allows for 256 discrete values, while 4-bit quantization allows for 16 values. This can be effectively represented with a histogram with each bin representing a single possible value.

In our case study, I use symmetric quantization to reduce the precision of the Relu output from float32 to int8, meaning, this representation allows for 256 discrete bins ranging from -127 to 127 values:

<br>

``` python
X = torch.randn(200, dtype=torch.float32)
X = F.relu(X)

bit_width = 8
alpha = torch.max(X)

# determine clipping range
r_max = 2 ** (bit_width - 1) - 1
r_min = -(2 ** (bit_width - 1) - 1)

# compute scaling factor S
scale = alpha / r_max

# symmetric quantization operation
quantized_data = X/scale
quantized_data = torch.clamp(torch.round(quantized_data), min=r_min, max=r_max).to(torch.int8)

plot_histogram(quantized_data, bit_width, quantized_data)
```

<br>


![](/assets/img/pexels/qlora/symmetric_quant.png)

{:.image-caption}
*Using symmetric quantization to convert the Relu output from float32 to int8 values, we select the largest absolute value in the data α, and create a clipping range of [ -α , α], then scale the values down  to map them to a quantized domain of [-127, 127]. This representation allows for 256 discrete bins ranging from -127 to 127 values. These quantized limits are depicted in the histogram by the blue and purple vertical lines.  [Link to code for converting to 8-bit via symmetric quantization and histogram plotting](https://gist.github.com/Manalelaidouni/f5243ad2a91ecb3c8602b87ac70ec1d1) — Created by Author.*

<br>

> We can clearly see in the histogram that half of the bins to the left of zero are not utilized at all, this leads to a more limited data representation, where instead of utilizing the entire 256 values assumed in an 8-bit representation, it only uses half of that, which is 128 values. This is equivalent to using a 7-bit format ($2^7=128$) resulting in one bit that is completely wasted out of the 8 bits.

<br>

However, if we use asymmetric quantization on Relu output, the full range of quantized values will be fully utilized, because the clipping range becomes $[r\_{min}=min(X),r\_{max}=max(X)]$ instead of $[ -α= - max(∣X∣) , α=max(∣X∣)]$ which includes the values that actually show up in the input data.

> To conclude, applying symmetric quantization to skewed data might allocate a significant portion of the quantized rangeto values that are unlikely to occur in practice, which makes asymmetric quantization more suitable for this type of data.

<br>

### *Some background about non-uniform quantization*

Now that you’ve reached this part of the post, I should point out to you that all of the previously mentioned information about quantization relates to linear quantization, otherwise reffered to as uniform quantization, which involves linear scaling with $S$ and results in equally spaced quantized values with a constant step size.

<br>

> The constraint of using equally spaced quantized values in uniform quantization is that it creates a representation that may not approximate the distribution of the input data well, increasing the performance gap between the full precision and its quantized counterpart.

<br>

That’s why non-uniform quantization also exists. Non-uniform quantization uses a non-linear function which generates values that are not necesseraly equally spaced, i.e, it can use non-constant steps resulting in different quantized levels.

<br>

> As such, non-uniform quantization with the same number of bits can create a representation that is more flexible to approximate the distribution of full precision values better, thus reducing the quantization error, leading to a better performance compared to uniform quantization.

<br>

For instance, neural network weights typically follow a Gaussian distribution with most of the values clustered around 0. In this case, applying non-uniform quantization will better represent the distribution of the weights. For example, we can use a non-uniform quantization method with increased resolution or smaller step sizes around 0.

<br>

> However, the challenge with non-uniform quantization is that using non-linear function involves look-up tables and additional logic that is not hardware efficient. Although a lot of recent work has been proposed to address this issue, one example is [Power-Of-Two quantization](https://arxiv.org/abs/2203.05025), which uses a base-2 logarithmic to quantize the numbers into powers of 2, allowing the dot-product to be accomplished using bit shifting instead of multiplication, which is more hardware friendly.

<br>

Consequently, uniform (linear) quantization is still considered the standard in most libraries that provide quantization support like Nvidia’s [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/precision.md), HuggingFace’s [Quanto](https://huggingface.co/docs/transformers/main/en/quantization#quanto), and PyTorch's built-in quantization module, mainly due to its simpler implementation.

<br>
<br>

# **GPU Memory Estimation**

Finetuning large models or simply running inference requires substantial GPU memory because it requires the model weights and intermediate computations to be loaded into the GPU’s memory.

<br>

This memory bottleneck is fundamentally imposed by the von Neumann architecture in current computing systems, where the main system memory is separated from the compute processing units on GPU. Therefore, all data required for neural network computations must be transfered from main memory to GPU memory (VRAM). This data in question includes the model weights, the gradients, the activations and optimizer states, all need to fit into the GPU memory.

<br>

Furthermore, the bottleneck is due to memory hierarchy, where memory component in computing systems can be arranged hierarchically in terms of storage, speed and cost. With the main memory being relatively cheap yet offereing larger storage with slower access to data, while GPU memory being expensive with smaller storage but offereing faster access speeds.

<br>

> To give you an idea of how much GPU memory is required for finetuning, a 7 billion parameter model loaded in 16-bit precision format, means each parameter is represented by 2 bytes of memory, thus you need 14 GB of GPU memory just to load the model weights alone. For a 32-bit precision, where each parameter is represented by 4 bytes, you would need 28 GB of GPU memory. Additionally, you have to account for some memory overhead, typically extra GBs to store forward activations, gradients and optimization states.

<br>

Memory requirement can be estimed using the following formula :


 $VRAM\_requirement = total\_number\_of\_parameters\ * \ number\_of\_bytes\_per\_parameter$

<br>
<br>


<a name="lora"></a>

# **LoRa Overview**

> As we mentioned above, finetuning language models is computationally expensive, almost prohibitive when dealing with LLMs that contain parameters in the order of tens of billions or more. As a result, many memory-efficient finetuning methods have been developed, one of which is LoRa (Low-Rank Adaptation of Large Language Models) which is introduced as a parameter-efficient finetuning approach that enables finetuning large models on resource-constrained devices, making it possible to adapt powerful language models to downstream tasks for everyone.

<br>

The LoRa paper states the following “*LoRa expresses the weight updates ∆W as a low-rank decomposition of the weight matrix*”. Let’s break this down to fully understand what does this mean:

<br>

First, let’s understand the terms, mathematically, the rank of a matrix refers to the number of linearly independent vectors in a matrix. Linearly independent vectors are vectors that can not be produced by linearly combining other vectors. As a result, they are considered the most important ones in the matrix that can not be constructed from others vectors and contain unique information. With this in mind, for an $m$ x $n$ matrix $A$, if the rank of this matrix is less than the number of its columns or rows, $rank (A) < min(m, n)$, we say that A is *rank-deficient* or has a *low-rank*.

<br>

The Low-Rank Adaptation paper is influenced by ideas in the [Li et al.](https://arxiv.org/pdf/2012.13255.pdf) and [Aghajanyan et al.](https://arxiv.org/pdf/1804.08838.pdf) papers that show that language models are over-parameterized and have a very low intrinsic dimension that can be effectively represented using fewer dimensions.

<br>

> It turns out that LLMs have a low dimension rank structure, meaning that important features and patterns in the model can be represented using significantly fewer dimensions, efficitively reducing the model complexity.

<br>



Based on these findings, the authors hypothesized that the finetuned part  $ΔW$, which is the change in weights during model adaptation — *where the weights are adapted to the new task* — also has a low intrinsic rank. The change in the weights  $ΔW$ refer to the the difference between the finetuned parameters $W_{finetuned}$ and the pre-finetuned parameters $W_{pretrained}$ in a single layer. 

<br>

![](/assets/img/pexels/qlora/lora_formula.png)



$∆W$ having a low rank implies that it can be decomposed into two smaller matrices $A$ and $B$ (hence low-rank decomposition) which are then adapted to the new dataset.

<br>

> Instead of learning the highly dimensional weight update matrix $ΔW$ like in regular full finetuning, LoRa approximates $ΔW$ with the product of two significantly smaller matrices and learn their parameters instead.

<br>

To examine this closely, a $d$ x $k$ matrix $ΔW$ can be factored as $ΔW=BA$, where $A$ is a $d$ x $r$ matrix and $B$ is a $r$ x $k$ matrix, where $r$ denotes the LoRa rank with $r << min(d, k)$. Using this approach, the total number of parameters becomes $r(d+k)$ instead of $d$ x $k$.

<br>

For instance, using a linear layer with a $1000$ x $1000$ weight matrix and a rank value of $r=3$, with LoRa, the number of trainable parameters for this layer would be reduced from $1,000,000$ to $6,000$. This is effectively reducing $99.4$% of the parameters in a single layer!

<br>

> *Note that LoRa rank `r` is a hyperparameter that determines the shape of the A and B matrices and evidently, selecting a smaller r value results in far fewer parameters to learn during finetuning, however it might not capture all the knowledge in the new dataset.*

<br>

Now that we understand that LoRa replaces the weight update matrix $ΔW$ of selected layers with much smaller matrices A and B, <i>**how are these matrices created?**</i>

<br>

Initially, the paper mentions that the parameters of matrix $A$ are initialized from a random Gaussian distribution while the parameters of matrix $B$ are initialized with zeros, this means that $BA$ is equal to zero intially before being updated during finetuning.



Furthermore, $ΔW$ is scaled by the factor of $\dfrac{α}{r}$ , where $r$ is the rank and $α$ is a hyperparameter needed for scaling.

<br>

> The objective of this scaling is to reduce the risk of adapters overfitting the data when using higher rank value and to maintain a consistent learning behavior for different rank values.

<br>

*Here is a rough code example, but for a detailed description of the implementation please refer to this [section](#4-injecting-lora-trainable-adapters)*.


<br>

``` python
class LoRaAdapter(nn.Module):
    def __init__(self, rank, alpha, d, k):
        super().__init__()
        self.scale_factor = alpha/rank
        # assuming weight matrix is of (d x k) dimensions
        self.A_matrix = nn.Parameter(torch.randn(d, rank))
        self.B_matrix = nn.Parameter(torch.zeros(rank, k))

    def forward(self, x, W_0):
        W_0.requires_grad = False
        result = W_0 @ x + ((self.A_matrix @ self.B_matrix) @ x) * self.scale_factor
        return result
```

<br>

> Only $A$ and $B$ are updated while the original weight matrix $W_0$ (the 4-bit quantized) remains frozen. This implies that the gradients and the parameter updates in the backward pass are computed only for A and B, resulting in a significant reduction in memory space used to store the checkpoints during finetuning. As a result, you’ll find that LoRA produces checkpoints of about few MB, unlike the memory intensive full finetuning that typically causes OOM errors. Furthermore, LoRa leads to a substantial reduction in computational costs and massive speed-up in the fine-tuning process since it focuses on a smaller set of parameters instead of updating the billion parameters in the LLM, all while producing a model with a comparable performance to full finetuning.

<br>


*Don’t take my word for it, here is the empirical results from the paper:*



The paper uses GPT-3 175B LLM as an example. Using LoRa finetuning, the number of trainable parameters was reduced by 10,000 times compared to full finetuning. Additionally, VRAM consumption during finetuning decreased from 1.2TB to 350GB, while checkpoint size was downsized from 350GB to 35MB. This translated into a 25% speedup on the finetuning process.

<br>

> To learn more about the training process with LoRa adapters and merging process, you can check this section of the post.

<br>


What’s unique about LoRa is that it doesn’t have any inference latency cost. After finetuning, the adapted weight matrices are simply combined with the original parameters. This is exceptionally useful when switching between tasks. That is, you can use the same base model finetuned on different downstream tasks and simply swap LoRa adapters for different tasks, as opposed to reloading multiple large models with the same large size, which is ofcourse memory and time inefficient.

<br>

> Using LoRa, task switching becomes more efficient because it consists of loading the weights of the base model one time and simply swapping the LoRa adapters for different tasks with minimal overhead instead of loading multiple finetuned models with the same large size.

<br>


Although LoRa adapters can be injected into any linear layer in the LLM architecture, in the paper, it exclusively injects the adapters into each of the projection matrices in the self-attention module, which are $W\_{query}$, $W\_{key}$, $W\_{value}$, and $W\_{output}$.

<br>


*You’ll gain a deeper understanding of LoRa in this [section](#4-injecting-lora-trainable-adapters) (part 4, 5, 7).*

<br>

# Technical Deep Dive into Implementing QLoRa

> Now that we have a solid theoritical understanding of how the LoRa finetuning approach works and some background on quantization, we can delve into how QLoRa combines the two concepts to dramatically reduce GPU memory usage when finetuning large models.

<br>


This section aims to be a technical deep dive into implementing QLoRa. Libraries involved in applying QLoRa:

* 4-bit quantization is implemented in the `bitsandbytes` quantization library created by Tim Dettmers which also offers 8-bit quantization among other techniques like 8-bit optimizers.
* HuggingFace 🤗 team provides LoRa finetuning via `PEFT` library (which provides several parameter-efficient finetuning methods) which is integrated with the Transformers library.
* The HuggingFace 🤗 team have [integrated](https://huggingface.co/blog/4bit-transformers-bitsandbytes) `bitsandbytes` with the 🤗 ecosystem, to enable finetuning methods available in `PEFT` to be applied on quantized (4-bit or 8-bit) base models with very few lines of code.

<br>
<br>

## **1\. Breakdown of 4\-Bit Quantization Using NF4 Data Type**


> `bitsandbytes` provides 4-bit quantization using block-wise k-bit quantization, this is applied to linear layers of the model through the `Linear4bit` class, which converts linear modules from `nn.Linear` to a `bnb.nn.Linear4Bit` layers.

<br>

Let's walk through the steps that a weight matrix $W$ goes through to be converted to the NF4 data type using block-wise quantization:

1. Flatten the weight matrix $W$ into a one-dimensional sequence.

2. Then split $W$ into equal sized blocks.

3. Normalize each block with its absolute maximum value to make sure the weights fit within the quantization range of [-1, 1].

<br>

> *This normalization is the scaling operation where the scaling factor S is the absolute maximum value, as a result the scaling factor of each block is stored for the dequantization process as a tensor with a length equal to the number of blocks.*

<br>

4. The actual quantization mapping uses a set of predefined unique 16 float values [$q\_1, . . . , q\_{16}$] suggested by the paper to map each value normalized value $x\_i$ in the block to the nearest quantized value $q\_i$ in the set. These `NF4_quant_levels` are further referenced in both the [paper](https://arxiv.org/pdf/2305.14314) (in Appendix E) and [the](https://github.com/TimDettmers/bitsandbytes/blob/1f2ca43ae5f3b453ff5fed73a17c661dc4fbbcb3/bitsandbytes/functional.py#L1087) [code](https://github.com/TimDettmers/bitsandbytes/blob/ffd7d0db6a660c97b60a2c9605309ee4b5cd40e3/csrc/kernels.cu#L3319).


``` python
NF4_quant_levels = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
```
<br>

<i>So why these `NF4_levels` values in particular?</i>The NF4 data type introduced in the [QLoRa](https://arxiv.org/abs/2305.14314) paper uses *4-bit quantile quantization* to generate a set of quantization levels which are then normalized to fit a range of [-1, 1]. This NF4 levels representation is best suited for normally distributed data with high precision around zero, which is the case for neural network weights which mostly follow Gaussian distribution, hence the name “*Normal Float*”.

<br>

**k-bit quantile quantization** estimates $2^b$ equally spaced quantile values $q\_i$ where $b$ is the desired bit-width for quantization. These estimated quantiles represent the quantization levels based on the quantiles of the normal distribution. To achieve this, the `bitsandbites` library uses the quantile function — also referred to as the inverse cumulative distribution function (ICDF) — implemented using `scipy.stats.norm.ppf` to generate these values. These quantiles are then normalized by the maximum value to achieve a range of [-1, 1]. Furthermore, Qlora uses asymmetric quantization to quantize $0$ in the input tensor to a constant non-zero value Z. This results in exactly $2^{b-1} + 1$ positive values and $2^{b-1}$ negative values in the representation (this makes it 9 positive values and 8 negative values in `NF4_quant_levels`).

<br>

**Now that we know from where these value come from**, the `bitsandbites` library uses midpoints between NF4 values as bins and compares each input value $x\_i$ to determine which bin the input value $x\_i$ falls into. It then assigns $x\_i$ to the closest $q\_i$ that falls within the bin.

<br>

> **TL;DR :** NF4 uses a mapping table of 16 levels `NF4_quant_levels` to map each input value to the nearest NF4 value. These are quantiles of a theoritical normal distribution that are normalized to fit a range of [-1, 1] to make them compact around 0. This means, the NF4 quantization is an asymmetric, non-uniform quantization scheme that is applied independently to each block of the input weights . These produced floating points quantization levels are ideal for normally distributed data like neural network weights with high precision around zero.

<br><br>



*The following steps 5 and 6 contain additional logic to store the final quantized values:*

5. Each quantization level is represented with a 4 bits binary value as referenced in this [part of the code](https://github.com/TimDettmers/bitsandbytes/blob/1f2ca43ae5f3b453ff5fed73a17c661dc4fbbcb3/csrc/kernels.cu#L278). The binary representations are used for storage are in unsigned 4-bit integers format (`uint4`), we will see at end of step 6 why we use the `uint4` representation. For instance, this [function](https://github.com/TimDettmers/bitsandbytes/blob/1f2ca43ae5f3b453ff5fed73a17c661dc4fbbcb3/csrc/kernels.cu#L223) uses these binary values to map back to the quantization levels during the dequantization process.

<br>

``` python
NF4_quant_4bit = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111]
# decimal representation of the above uint4 binary values:
NF4_quant_4bit = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
```
<br>

6. Finally, pack the quantized 4-bit tensor into `torch.uint8` dtype (an *8-bit unsigned integer representation*).

<br>

> Admittedly, this part intially threw me off, as I was expecting the 4-bit representation to be packed into a 4-bit data type which assumes exactly 16 unique values, not an 8-bit data type with 256 unique values. However, after going through the code, it turns out the author of `bitsandbytes` converts the 4-bit values into 8-bit by packing two 4 bit values into a single 8-bit value, this results ofcourse, in a different shape for the quantized tensor. This is because PyTorch does not support 4-bit data types and the smallest type it supports is 8-bits — as of the writing of this post — Furthermore, the reason it uses an 8-bit integer format and not an 8-bit floating point format (FP8) is due to the lack of native support for FP8 in PyTorch. The packing operation is exactly what Pytorch’s new data type “*quantized 4-bit integer*” `torch.quint4x2` does as well, as you can see in the [documentation](https://pytorch.org/docs/stable/tensors.html#id11). The packing of two 4-bits values to 8 bits is very straightforward using simple bitwise operations. The actual packing step in `bitsandbytes` is performed in this [part of the code](https://github.com/TimDettmers/bitsandbytes/blob/1f2ca43ae5f3b453ff5fed73a17c661dc4fbbcb3/csrc/kernels.cu#L819), but make sure to follow along to see our implementation.

<br>
<br>

*To make sure my understanding of the 4-bits quantization with the NF4 data type (described in the above steps) is accurate, I implemented it from scratch on a small dummy input (first code snippet) and compared it to the `bitsandbytes` implementation (second code). Thankfully, the produced values are exactly equal. Note that the smallest block size used in `bitsandbytes` is 64 elements and since my input $W$ has 20 elements, `bitsandbytes` treats the entire tensor as a single block, so I didn't split $W$ into smaller blocks.*

<br>

``` python
import torch

NF4_quant_levels = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
nf4_quant_4bit = torch.tensor([0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111])

# generated tensor of [5, 4] shape
W = torch.tensor([[ 0.4767, -0.2921,  0.0787, -0.1018],
                  [-0.3453,  0.3834, -0.0107, -0.4692],
                  [-0.4072, -0.2996, -0.4942, -0.2640],
                  [ 0.0125,  0.2962,  0.3123, -0.4705],
                  [-0.1982, -0.1545,  0.3358, -0.4086]])

# flatten the tensor
flat_W = W.flatten()

# normalize the tensor using absmax to fit within [-1, 1]
max_val = flat_W.abs().max()
normalized_W = flat_W / max_val

# map each input value to its nearest quantization level - then to its 4-bit binary representation
quantized_W_4bits = torch.zeros(normalized_W.shape, dtype=torch.int)
for i, val in enumerate(normalized_W):
    closest_level = torch.argmin(torch.abs(NF4_quant_levels - val)) # get the index of closest quantization level
    quantized_W_4bits[i] = nf4_quant_4bit[closest_level]

print(quantized_W_4bits) # [15,  2,  9,  5,  1, 14,  7,  0,  1,  2,  0,  2,  7, 13, 13,  0,  3,  4, 14,  1]

packed_W_8bits = []
for i in range(0, len(quantized_W_4bits), 2):
    # Courtesy of https://www.geeksforgeeks.org/store-two-numbers-in-one-byte-using-bit-manipulation/
    # take each pair of 4-bit values in quantized_W_4bits and combine them into packed_W_8bits by shifting
    # the first 4 bits to the left using the left shift operator '<<' and combining them using the OR | operation
    result = (quantized_W_4bits[i] << 4) & 0xff
    result =  result | quantized_W_4bits[i + 1]
    packed_W_8bits.append(result)

# set it as torch.uint8
packed_W_8bits = torch.tensor(packed_W_8bits, dtype=torch.uint8).view(-1, 1)

print(packed_W_8bits)
""" Output:
tensor([[242],
        [149],
        [ 30],
        [112],
        [ 18],
        [  2],
        [125],
        [208],
        [ 52],
        [225]], dtype=torch.uint8)
"""
```

<br>

> To show you how packing the first pair of `quantized_W_4bits` (15, 2) results in 242 in `packed_W_8bits`: packing is a series of bitwise operations, therefore 15 is represented as `1111` in binary, while 2 is is represented as `0010`. Following the packing operation mentioned above `(1111 << 4) | 0010 = 11110000 | 0010 = 11110010` ; `11110010` is equal 242 in decimal.

<br>

*Here is a bitsandbytes code you can run if you want to check the output of 4-bit quantization with NF4 dtype yourself — it results in the same output.*

<br>

``` python
import bitsandbytes as bnb 
from bitsandbytes.nn import Linear4bit, Linear8bitLt
import torch

W = torch.tensor([[ 0.4767, -0.2921,  0.0787, -0.1018],
                  [-0.3453,  0.3834, -0.0107, -0.4692],
                  [-0.4072, -0.2996, -0.4942, -0.2640],
                  [ 0.0125,  0.2962,  0.3123, -0.4705],
                  [-0.1982, -0.1545,  0.3358, -0.4086]]).to(cuda)

print(W.weight.dtype) 
# Output: torch.float32 

# 4-bit quantization with NF4 datatype
quantized_W = bnb.nn.Linear4bit(
    W.in_features,
    W.out_features,
    bias=False,
    quant_type="nf4",
    device="cuda")

quantized_W.weight = bnb.nn.Params4bit(data=W.weight, requires_grad=False).to("cuda")

print(quantized_W.weight.dtype) 
# Output: torch.uint8 

print(next(quantized_W.parameters()))

""" Output:
Params4bit([[242],
            [149],
            [ 30],
            [112],
            [ 18],
            [  2],
            [125],
            [208],
            [ 52],
            [225]], device='cuda:0', dtype=torch.uint8))

"""
```

<br>


Moreover, the reason `bitsandbytes` uses an unsigned `uint8` for packing instead of signed integers `int8` is that `int8` can represent negative values by reserving the bit furthest to the left as the sign bit. However, during the packing step, the left shifting operation can become tricky when dealing with the sign bit. Using `uint8` removes this complexity, by keeping only the bits of the actual two 4-bit numbers without including the sign bit. That is why, in step 5 it uses the `uint4` format (ranging from 0 to 15) and not `int4` (ranging form -8 to 7) to represent each quatized 4-bit value.

<br>

> **TL;DR** : In step 6, it’s compacting two `uint4` numbers into a single `uint8` number. This implementation effictively reduces the model memory footprint to half of 8-bit format. Memory wise, this is equivalent to using a 4-bit quantization format. This is the current Pytorch implementation of any sub-8 quantization. For instance, in 2-bit quantization (given an efficient quantization mapping scheme with minimal quantization error) each value is mapped into a 2-bit value, then each four 2-bit values are packed into a single 8 bit value, achieving the same memory efficiency as a 2-bit quantization would.

<br>
<br>

## **2\. Loading and Configuring the NF4 Quantized Model**


*In this section, I explain the configuration choices necessary for implementing of 4-bit quantizated model, this is followed by a step by step process for implementing LoRa finetuning using this quantized model.*

<br>

> Up to this point, we have a model whose linear layers are quantized as NormalFloat 4-bit data type (NF4) and converted from `nn.Linear` to `bnb.nn.Linear4Bit`. Thanks to the integration of the two libraries (`bitsandbytes` and 🤗’s `Transformers`), it only takes one step to load the model and quantize it, by enabling `load_in_4bit` flag and setting `bnb_4bit_quant_type` to NF4 using `BitsAndBytesConfig` from `Transformer` as follows.

<br>

``` python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
   bnb_4bit_use_double_quant=True)

# *Using Open LLama 3b model*
base_NF4_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b",
																											 quantization_config=nf4_config)

print(base_NF4_model.is_loaded_in_4bit) 
# Output: True
```

<br>

### *What does `bnb_4bit_compute_dtype` do?*

In this step, we can specify the data type to use for computation via the `bnb_4bit_compute_dtype` argument. This determines the desired target dtype for the dequantization process (regardless of the dtype used for the input), you’ll see its use in this [section](#7-merging-lora-adapters-to-base-model).

<br>

Furthermore, you can set `bnb_4bit_compute_dtype` to `float32` (the default) or `bfloat16` (16-bit BrainFloat), if you’re not familiar with with <b>`bfloat16`</b>, this is a great opportunity to learn about it as it’s commonly used in deep learning. Here is what you need to know:

<br>

> **16-bit BrainFloat** is a floating point representation developed by Google Brain tailored for deep learning models. This representation has the same number of bits as `float16` but with a different allocation of bits, assigning 8 bits for the exponent as opposed to 5 bits in `float16`. The increased exponent width offers a larger dynamic range, meaning it can represent a larger range of values (dynamic range is the ratio of the largest to smallest positive value). As a result, `bfloat16` has a comparabale performance as `float32` with half the number of bits. This offers more flexibility to represent smaller and larger values in the network, reducing numerical underflows and overflows all while using less memory during training.

<br>

Therefore, it’s recommended to use `bfloat16` as `bnb_4bit_compute_dtype` to perform operations. Consequently, the weights will be dequantized from 4-bits in `NF4` to 16-bits in `bfloat16` dtype.

<br>

### *What does `bnb_4bit_use_double_quant` do?*

At the normalization step of block-wise quantization, we mentioned that the scaling factors, which are the absolute maximum values, are stored for dequantization. These scaling factors are essentially quantization constants that require additional space to be stored. To address this, the QLoRa paper suggests a solution to further reduce memory storage, which is to quantize the quantization constants themselves using block-wise quantization. This way, we end up with both quantized weights and their corresponding quantization constants being quantized as well. This is called **double quantization** (or nested quantization). According to the paper, double Quantization saves an average of about 0.37 bits per parameter, approximately 3 GB for a 65B model.

<br>

## 3\. Casting Certain Layers to Full Precision


After loading the quantized base model, it’s recommended to run the following line of code:



``` python
base_NF4_model = prepare_model_for_kbit_training(base_NF4_model)
```
<br>

***What does it actually do?***

After quantizing all of the linear layers in the model, it’s important to cast some layers back to `float32` (fp32) for a better numerical stability. This function casts some layers to `float32`. Specifically, it upcast any normalization layer (*BatchNormalization* or *LayerNormalization*) to fp32 because they contain reduction operations like sum and average, which can cause NaN loss values when performed with reduced precision. Therefore, to maintain as much information as possible, we use higher number of bits.



In addition to normalization layers, the function also upcasts the embedding and LM head layers. Furthermore, it freezes the entire model and enables gradient checkpointing, which we will discuss [later](#gradient-checkpointing--saving-gpu-memory-by-storing-less-activations) in the post.

<br>

Inspecting the model before calling `prepare_model_for_kbit_training` :

``` python
# normalization - embedding - LM head layers
print(base_NF4_model.model.norm.weight.dtype, base_NF4_model.lm_head.weight.dtype, base_NF4_model.model.embed_tokens.weight.dtype)
# Output : (torch.float16, torch.float16, torch.float16)

print(base_NF4_model.is_gradient_checkpointing)
# Output : False
```

<br>

Inspecting the model after calling `prepare_model_for_kbit_training` :

``` python
print(base_NF4_model.model.norm.weight.dtype, base_NF4_model.lm_head.weight.dtype, base_NF4_model.model.embed_tokens.weight.dtype)
# Output : (torch.float32, torch.float32, torch.float32)

print(base_NF4_model.is_gradient_checkpointing)
# Output : True
```

<br>



## 4\. Injecting LoRa Trainable Adapters

Next, we setup `LoraConfig`, where we specify the layers to inject the loRa adapters into, using `target_module`. Alternatively you can select all linear layers instead of just few by setting `target_modules="all-linear"`. *Following my example I specify the attention and MLP modules in `target_modules`.*

<br>

This is also where we define the rank dimension, which determines the shape of the adapter matrices, as well as the Lora dropout, which we mention its use at this [section](#rank).

<br>

``` python
target_modules =  ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear", 
    lora_dropout=0.1, 
    task_type="CAUSAL_LM",
    bias="none")
```

<br>

Next, we wrap the configuration `lora_config` with `get_peft_model` to create a trainable `PeftModel` :



``` python
peft_model = get_peft_model(base_NF4_model, lora_config)
```

<br>

***But what does this line of code actullay do ?***


1. It creates the adapters, we’ve mentioned earlier that `LoRa_A` is initialized from a Gaussian distribution, however the PEFT library initializes the matrix from [Kaiming-uniform](https://github.com/huggingface/peft/blob/5a4b9cade64bac8afdff5006ee9dd815c90b5469/src/peft/tuners/lora/layer.py#L141) distribution while initializing the `LoRa_B` matrix with zeros.
2. It then injects these matrices into the layers defined in `target_modules`.
3. It makes these adapter weights trainable while keeping the rest of the quantized model frozen.

<br>

## 5\. Finetuning Process with LoRa

*How does the forward & backward phase performed with LoRa adapters?*


<br>

![ Illustration showing part of forward pass of input X in layer $W_Q$ which we selected to inject loRa adapters into (layer specified in `target_modules`). This implementation is specific to the QLoRa paper, not the LoRa paper — Illustration Created by Author.](/assets/img/pexels/qlora/loRa_forward.png)

{:.image-caption}
*Illustration showing part of forward pass of input X in layer $W_Q$ which we selected to inject loRa adapters into (layer specified in `target_modules`). This implementation is specific to the QLoRa paper, not the LoRa paper — Illustration Created by Author.*


<br>

* As you can see above, the trainable weight matrices $A$ and $B$ (which constitute $ΔW$) are simply injected into the existing model layer $W\_Q$, as opposed to another existing [solution](https://arxiv.org/pdf/1902.00751) that also uses adapters but adds them as separate layers sequentially to the Transformer architecture, introducing additional inference latency. The layer weight itself $W\_Q$ is kept frozen.

<a name="rank"></a>
* LoRa dropout introduced in the QloRa paper and is applied to the input $X$ before multiplying by the matrices $A$ and $B$. You can set the value of the `lora_dropout` hyperparameter in `LoRa_config`.

<br>

![*Inspecting `W_Q`  referred to  in the model architecture as `q_proj`, after injecting the adapters into it. Using OpenLLama as quantized base  model (hence `Linear4bit` class) with a LoRa rank of 8 and a dropout of 0.1 —* Illustration Created by Author.](/assets/img/pexels/qlora/Lora_code.png)
{:.image-caption}
*Inspecting `W_Q`  referred to  in the model architecture as `q_proj`, after injecting the adapters into it. Using OpenLLama as quantized base  model (hence `Linear4bit` class) with a LoRa rank of 8 and a dropout of 0.1 —* Illustration Created by Author.*

<br>

> This means, ofcourse, that in the backward pass only a smaller set of parameters are updated, that is, fewer gradients to compute and much less optimizer states to maintain compared to full fine-tuning. This leads to faster, more affordable finetuning with less storage requirements.

<br>

## 6\. Strategies for Reducing GPU Memory Usage in LLM Fine\-tuning with STTrainer



Besides quantization and LoRa’s efficient finetuning, which are used as memory saving approaches, there are other methods that are currently prevelant in training LLMs. These methods reduce memory consumption without compromising performance and are related to storing activations, gradients and optimizer states during finetuning.

<br>

> *We can easily enable the following techniques using the `TrainingArguments` class from `Transformers`, whose object is then passed to Supervised Fine-tuning Trainer `SFTTrainer` from `TRL` library.*

<br>

*The configutation in the following code contains only the settings that pertain to the important training techniques explained in this section for demonstration purposes, it omits necessary configs related to training like batch\_size, learning\_rate, etc. Here is a comprehensive [code of SFTTrainer](https://gist.github.com/younesbelkada/f48af54c74ba6a39a7ae4fd777e72fe8) with complete configuration.*

<br>

``` python
training_args =  TrainingArguments(output_dir=output_dir,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    gradient_accumulation_steps = 4)
															
trainer = SFTTrainer(peft_model,
    args = training_args,
    peft_config = lora_config,
    train_dataset = train_data,
    dataset_text_field="input",
    tokenizer=tokenizer)
    		
trainer.train()
```

<br>


### ***Gradient checkpointing — Saving GPU memory by storing less activations.***

Gradient checkpointing is a technique used to reduce model memory footprint during backpropagation by storing less activations. Typically in vanilla neural network training, the activations in the forward pass are cached so they can be used in the backward pass to compute the gradients. However for a large model, caching all activations can cause memory overhead. Gradient checkpointing is used to address this issue.

<br>

> The crux of gradient checkpointing is that instead of storing all activations, we store only a subset of them in the forward pass, mainly the high cost operations that are time consuming, while the remaining ones are re-computed on the fly during backpropagation. This reduces memory usage but it takes longer to complete a backward pass.

<br>

I believe the name “*gradient checkpointing*” is somewhat misleading, its alternative name, “*Activation checkpointing”* seems more appropriate. You can enable gradient checkpointing by setting `gradient_checkpointing` to True in `TrainingArguments`.

<br>

### ***Paged optimizer — Saving GPU memory during fine-tuning by moving optimizer states to CPU memory only when GPU memory is fully saturated to avoid out of memory error.***


> Optimizer states take up a lot of GPU memory, sometimes more than model weights when using stateful optimizers (unlike stateless optimizers like SGD). For instance `AdamW` maintains 2 states per parameter (the first and second moments of the gradient) that need to be stored in memory. This can quickly lead to memory spikes during finetuning. To overcome this issue QLoRa paper introduces paged optimizers.

<br>

***How does it work under the hood* ?**

Paged optimizers use automatic page migration of optimizer states from GPU memory to CPU memory to avoid running out of GPU memory and crash.This technique uses the Unified Memory model introduced in Cuda toolkit 6.0, which provides a single memory space that is directly accessible by CPU and GPU. This technology allows data transfer (page migration) to be automatic in the background using the Page Migration Engine, as opposed to GPU offloading which requires developers to use explicit copy functions to migrate pages between GPU and CPU which is error prone. This effectively prevents OOM error during LLM finetuning. For further [reading](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-in-cuda/) on Unified Memory with Cuda.

To implement this we set the `optim` argument to `paged_adamw_32bit` in `TrainingArguments` class.

<br>

### ***Gradient Accumulation — Saving GPU memory by performing less parameter updates.***



Typically in neural network training, a parameter update is performed for each batch. In gradient accumulation, parameter update is performed after multiple batches, this means gradients are accumulated (summed up) for a number of batches before being used to perform a parameter update. However to ensure the magnitude of gradients
is within a reasonable range and not too large due to summation, we normalize the loss with the number of accumulation steps.

<br>

> **Furthermore, gradient accumulation simulates training with a larger batch size while actually using smaller batch size**. This is because using accumulated gradients over many smaller batches provides more accurate gradient information, which improves the direction of the parameter updates and speeds up convergence. This is similar to what would be obtained from a single large batch, all while avoiding memory constraints that arise with using large batch sizes.


<br>

You can enable gradient accumulation by setting the `gradient_accumulation_steps` argument to greater than 1.

<br>

## 7\. Merging LoRa adapters to base model


Once the finetuning process is complete, we merge the stored fine-tuned LoRA adapters — saved in safetensors format and taking up very few MB of space — back into the original layers specified in `target_modules` of the quantized base model `base_NF4_model` that we used at the start, to produce the final model for inference. This is achieved using the following code:

<br>

``` python
# base_dir specifies the directory where the fine-tuned LoRA adapters are stored
lora_model = PeftModel.from_pretrained(model=base_NF4_model, model_id=base_dir)
merged_final_model = lora_model.merge_and_unload()
```

<br>

***What does it do?***

For each module specified in `target_modules`, the finetuned LoRa adapters are added to their corresponding original quantized weights.

For instance, $W\_Q$ is updated with the following formula $W\_Q$ = $W\_Q$ + $ΔW$.

<br>

> It's important to note that we dequantize the original weight matrix $W\_Q$ from `NF4` to `bfloat16` — the computation dtype we specified in `BitsAndBytesConfig` — before merging. Dequantization is necessary here to perform the addition operation in higher precision format and maintain as much accuracy as possible.

<br>

Here are the exact merging steps :

1. Compute $ΔW$ for $W\_Q$ using its corresponding finetuned LoRa weights `A` and `B` as follows:
    $ΔW = A  *  B  * scale\_factor$
    where $scale\_factor=\dfrac{α}{r}$ as mentioned in the LoRa overview [section](#lora).

2. Dequantize the original weight matrix $W\_Q$ from `NF4` to `bfloat16`.

3. Add the LoRA adapters to the dequantized weight :
$dequantized\_W\_Q$ = $dequantized\_W\_Q$+ ΔW

4. Quantize the merged weight back to the original 4-bit quantization with NF4 dtype.

This process ensures that each of the LoRa adapters are integrated into the quantized base model weights efficiently.

<br>

# **LLM Inference Process**

Now that we have our quantized models finetuned on our dataset of choice, we can effectively run inference on it (like any other 🤗 model) using the `generate()` method from 🤗 to produce an output as follows:

<br>

``` python
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
with torch.no_grad():
	outputs = merged_final_model.generate(input_ids=inputs['input_ids'])
response = tokenizer.decode(outputs[0])
```

<br>

> *I bet you guessed the next question : but what is actually happening during inference to produce our desired response?* To answer this question, this section describes the inference process which takes in a prompt as input and generates a response. LLM inference consists of two phases, prefill and decoding :

<br>

### ***Prefill Phase of Inference***


During the prefill phase, the entire prompt tokens are fed to the model at once and processed in parallel until the first output token is generated.

<br>

It’s important to note that we are using a decoder-only based transformer (most common architecture like the GPT model used for next token prediction task) as our code example and the following figure reflects that. Transformer models contain many blocks, but the most important component of the architecture is the masked multi-head causal attention mechanism, which we use during the pre-fill phase. We use masking in this phase (just like in training stage), because we process the entire input sequence at once and we want to prevent the decoder from attending to future tokens and only consider previous ones up to the current token. The figure below summarizes this step:

<br>

![The figure visualizes single head attention with a single sample (not a batch) as input for simplicity. However, transformer-based models currently use multi-head attention (MHA), which allows multiple heads to capture different representations of relationships within the input sequence, resulting in a final representation with a strong contextual understanding of the input — Illustration Created by Author.](/assets/img/pexels/qlora/attention.png)

{:.image-caption}
*The figure visualizes single head attention with a single sample (not a batch) as input for simplicity. However, transformer-based models currently use multi-head attention (MHA), which allows multiple heads to capture different representations of relationships within the input sequence, resulting in a final representation with a strong contextual understanding of the input — Illustration Created by Author.*


<br>

> If you're wondering how multi-head attention (MHA) differs from the single-head attention shown in the figure, here's a short overview: MHA splits the generated Q, K, and V matrices with an assumed size of [batch\_size, seq\_len, embed\_dim] along the `embed_dim` dimension into `num_heads` parts. This way, each head gets its own set of $Q$, $K$, and $V$ matrices of shape [batch\_size, seq\_len, d\_k] with `d_k = embed_dim/num_heads` (hence the scaling by d\_k in the operation).

<br>

> However, in practice we use reshaping instead of splitting to apply the scaled dot product (i.e., the attention operation) on a single matrix so that the computation becomes efficiently parallelized. The reshaping is followed by a simple dimension permutation to get $Q$, $K$, and $V$ to have the [batch\_size, num\_heads, seq\_len, d\_k] dimension. This way, the same attention operation is computed independently (just like in the figure) to get the weighted sum of values from each head ($Y\_1$, $Y\_2$, ..., $Y\_h$). The final concatenation of the results from all heads is just another tensor reshaping followed by a linear transformation to project the combined context vectors back to the dimenstion of the input $X$ [batch\_size, seq\_length, embed\_dim].

<br>

> *Here is a corresponding implementation from Karpathy’s [mingpt](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L57), and here is great [post](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853) if you want to learn more about multi-head attention mechanism.*

<br>


### ***Decoding Phase of Inference***


The decoding phase contains multiple steps where the model generates tokens one at a time. At each decoding step, the model takes the user prompt coupled with the tokens predicted in previous steps and feeds them into the model to generate the next token.

<br>

> This phase showcases the autoregressiveness of the model, where it recursively feeds its own output back as input and thus predicting future values based on past values. This process continues until the model generates a special end token or reaches a user-defined condition, such as a maximum number of tokens.

<br>

***Zooming a bit into the attention block in this stage :***

At each decoding step, the new token is added to the input sequence, which includes both the user prompt and previously generated tokens. This newely added token attends to all the previously generated tokens through the multi-head attention block where the vector of attention scores are computed using a the scaled dot product.

> The scaled-dot product operation in this phase, constructs the query from the new token, while the keys and values are constructed from all tokens. Unlike the prefill phase, no masking is used in the decoding phase because future tokens are not known and have not been generated yet.

<br>

![GIF shows the reconstruction of K (key) and V (value) matrices using the query and value vectors of previous tokens along with those of the new token, with $x_4$ as the new token added to the input sequence [$x_1$, $x_2$, $x_3$]. This means the query vector of the new token will be multiplied with the key matrix K. The result of this dot-product (after scaling and applying softmax) is multiplied by the value matrix V —  Illustration Created by Author.](/assets/img/pexels/qlora/kv_gif.gif)

{:.image-caption}
*GIF shows the reconstruction of K (key) and V (value) matrices using the query and value vectors of previous tokens along with those of the new token, with $x_4$ as the new token added to the input sequence [$x_1$, $x_2$, $x_3$]. This means the query vector of the new token will be multiplied with the key matrix K. The result of this dot-product (after scaling and applying softmax) is multiplied by the value matrix V —  Illustration Created by Author.*

<br>

> The challenge that arises in the decoding phase is the need to recompute key and value vectors of all previous tokens at each step, resulting in redundant floating-point operations (FLOPs) and slowing down the decoding process.

<br>

### ***Optimizing decoding phase with KV Caching***

<br>

The solution to this is to cache the key and value vectors of each token once they are computed to avoid redundant computations. This cache is referred to as the KV-cache, which significantly speeds up the inference process. The KV-cache is first constructed during the prefill phase for all tokens in the prompt. Then at every decoding step, the cache is updated with the new key and value vectors of the newly generated token, making them the only vectors computed at this step.

<br>

![Example of using Q and V caches and computing only query, key and value vectors of the new token $x_4$ — Illustration Created by Author.](/assets/img/pexels/qlora/kv_cache.png)

{:.image-caption}
*Example of using $Q$ and $V$ caches and computing only query, key and value vectors of the new token $x_4$ — Illustration Created by Author.*

<br>

In practice, the KV-cache is just a tuple ($K\_cache$, $V\_cache$), where each has the shape $[batch\_size, seq\_len, num\_heads, head\_dim]$, meaning each attention head in the attention mechanism has a separate KV-cache for keys and values. Here is a pseudocode that demonstrates both the prefill and decoding phases using KV-caching and greedy search decoding for simplicity:

<br>

*Here is a pseudo-code of what would the inference process looks like:*

``` python
def run_inference(model, tokenizer, prompt, max_new_tokens):
    """
    LLM inference dummy function for a single sample.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    kvcache = (None, None)
    model_output_ids = []
    model.eval()

    with torch.no_grad():
        # Prefill phase: process the input prompt at once and populating the KV-cache
        # until the first output token is generated
        probabilites, kvcache = model(input_ids, kvcache=kvcache)
        next_token = torch.argmax(probabilites, dim=-1) # select the most probable token
        model_output_ids.append(next_token)

        # Decoding phase: generating tokens one by one
        model_output_ids.append(next_token)
        inputs_ids = torch.cat(inputs_ids, next_token)

        i = 0
        while i < max_new_token:
            probabilites, kvcache = model(input_ids, kvcache=kvcache)
            next_token = torch.argmax(probabilites, dim=-1)
            model_output_ids.append(next_token)
            
            # stop if next_ids is eos token
            if next_id == tokenizer.eos_token_id:
                break
                
            # continue decoding & add generated token to input for the next step			
            inputs_ids = torch.cat(inputs_ids, next_token)  # append prediction to input
            i += 1

    return tokenizer.decode(model_output_ids)
```

<br>

Note that we enable KV caching during inference in the `generate()` by enabling the `use_cache` flag.

<br>

# **Decoding Strategies**

Earlier, we mentioned the greedy search method for selecting the next token, but there are more effective approaches available. Initially, the language model generates a probability distribution over all possible tokens in the vocabulary, the method used to select the next token from this distribution depends on the chosen decoding strategy for text generation.

If you ever had to deal with language model APIs, you may have encountered these methods as configuration options to generate a desired output and you’ll also configure them in the *huggingface* `generate()` method we mentioned earlier. Generally, decoding strategies can be categorized into two main classes: *maximization based decoding* and *sampling based decoding*.

<br>

## *Maximization Based Decoding*

> *Maximization based decoding selects the next token with the highest probability, that is, it optimizes for model outputs that maximize likelihood. Here are some techniques*:

<br>

### Greedy Search

Greedy search is a simple, naive method that selects the most likely token at each decoding step to construct the final output sequence. However, because it considers only locally optimal tokens it results in low quality, incoherent text.

<br>

### Beam Search

Beam search is considered a more effective decoding strategy than greedy search, because it’s a search based algorithm that expands the search space at each time step. Instead of simply choosing the single best option at each step like in greedy search, beam search consistently updates of the top K sequences at each step, where K is referred to as the beam width. At the end of the decoding process, we end up with a set of K complete candidate sequences, from which we select the one with the highest probability. As a result, the beam search algorithm is a computationally intensive requiring more model runs than other decoding strategies, because it explores multiple paths by expanding over all selected top k tokens at each decoding step.

<br>

However, this decoding strategy performs poorly on open text generation tasks generating repetitive text even when setting a higher beam width.

<br>

> Beam search might perform well on some NLP tasks with a limited output such as machine translation because it requires a more constrained decoding process that prioritizes accuracy and exact outputs that closely resembles the input. However Beam search doesn’t perform as well on open ended text generation that require more variety in the output. [It has been found](https://arxiv.org/pdf/1904.09751) that language models assign higher probability to tokens with high grammaticality, which when generating longer text, tends to prioritize more generic tokens in a repetitive loop, yielding low quality output — this is referred to as neural text degeneration — Furthermore, human generated text is characterized by diversity, so to make text generation more human like, we use new techniques that introduce randomness into the decoding process to combine likelihood and diversity. This class of techniques is called sampling based decoding.

<br>

## *Sampling based decoding*

> *Sampling based decoding introduce some randomness into the decoding process, which results in a more diverse higher quality text generation. This class includes the following techniques* :

<br>


![Illustration explaining different sampling decoding methods —  Created by Author.](/assets/img/pexels/qlora/decoding_strategies.png)
{:.image-caption}
*Illustration explaining different sampling decoding methods —  Created by Author.*

<br>

### **Top-k Sampling**

<br>

Top-k Sampling introduces randomenss by considering a smaller set of tokens from which we randomly sample the next token from. The chosen tokens are the ones with the highest logit scores, which are unormalized model predictions. This way after applying softmax, the probability distribution is reconstructed and redistributed over this smaller set of candidates, it then samples the next token at random from this new distribution. This approach balances exploring a focused region of highly probable tokens with random sampling. The size of the chosen region is defined by the `top_k` hyperparameter. Here is the exact algorithm:

1. Select the k highest logits.
2. Set the smallest logit in the top-k logits as a threshold.
3. Keep only tokens with logit values higher than this threshold. In practice, we do not discard the tokens that are below the threshold, we actually assign them a highly negative value (- inf) to ensure that their produced probability after using softmax is exactly 0.

<br>


```python
def top_k_sampling(logits, top_k):
    """
    logits [batch_size, vocab_size] : unormalized logit scores.
    top_k : number of tokens to sample from.
    """
    top_logits, _ = torch.topk(logits, top_k, dim=-1) # [batch_size, top_k]
    smallest_in_top_k = torch.index_select(top_logits, 1, torch.tensor([2])) # [batch_size, 1] 
    logits[logits < smallest_in_top_k] =  -float('Inf')
    probs = F.softmax(logits, dim=-1) # [batch_size, vocab_size]
    sampled_token = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
    return sampled_token
```


<br>

> Although top-k sampling generates a relatively higher quality text with less repetition than beam search, its challenge lies in choosing an appropriate value for the hyperparameter k, because the choice of k depends mostly on the underlying distribution of logits, which varies ofcouse, from one decoding step to another. For instance, it wouldn’t be suitable to sample a small set of tokens from a flat distribution, nor would it be appropriate to sample a large set from a peaked distribution. As result, the following decoding method, “nucleus sampling” mitigates this issue by making the selection of k dynamic.

<br>

### **Nucleus sampling**

<br>


Nucleus sampling (also referred to as top-p sampling) Instead of having to select a k fixed number of tokens, nucleus sampling dynamically selects the set of tokens to sample from at each decoding step, based on a pre-determined probability threshold $top\_p$ which ranges from 0 to 1.

<br>

Nucleus sampling starts by determining the smallest set of tokens whose cumulative probability is at least $top\_p$ such that $\sum\_{i=1}^{k}p\_i ≥ top\_p$. This set of chosen tokens is analogous to a "nucleus” containing the most probable tokens. Once this set S={$𝑥\_1$,$𝑥\_2$ ,…, $𝑥\_𝑘$} is determined, it re-applies the softmax function to redistribute the probability over S, and a token is sampled at random from this new distribution (similar to what we did with top-k sampling).

<br>

Here is the exact algorithm for further understanding:

1. Apply softmax to generate probabilities over all tokens in the vocabulary.
2. Sort the tokens based on their probabilities in descending order.
3. Include tokens in the final set, starting from the top token with the highest probability, then add the tokens one by one until the cumulative probability of the tokens in this set is at least $top\_p$, that is, we should stop at the token $x\_k$ with a cumulative probability that first exceeds the threshold $top\_p$ . This means we include tokens $𝑥\_1$,$𝑥\_2$ ,…, $𝑥\_𝑘$ in the nucleus S.
4. Set the logits of the tokens not in the nucleus set *S* to −inf. This effectively removes these tokens from consideration in the next sampling step.
5. Re-apply the softmax function to the modified logits to get the new probability distribution over the nucleus set *S*.
6. Sample a token from the normalized distribution.


<br>


``` python
def nucleus_sampling(logits, top_p):
	probs = F.softmax(logits, dim=-1)
	ordered_probs, ordered_indices = torch.sort(probs, descending=True, dim=-1)
	cumulative_probs = torch.cumsum(ordered_probs, dim=-1) 
	
	# create mask of tokens whose cumulative probability is greater than top_p
	# all the False elements including the first True element
	mask = cumulative_probs > top_p
	
	# include the token that exceeds the threshold - index of the first True element to mask matrix
	indices_to_include = mask.float().argmax(dim=-1, keepdim=True)
	mask[torch.arange(mask.shape[0]).unsqueeze(1), indices_to_include] = False 
	
	# create the new mask with 0s and 1s
	new_mask = mask.float()
	
	# apply the mask to logits and replace masked out values (0.0) with -inf
	modified_logits = logits * (~ mask) #(1 - new_mask)
	modified_logits = modified_logits.masked_fill(modified_logits == 0.0, float('-inf'))
	
	# Map the modified logits back to the original order	
	template = torch.zeros(logits.shape)
	final_logits = template.scatter(1, ordered_indices, modified_logits)
		
	# repass the logits back to softmax
	probs = F.softmax(final_logits, dim=-1) # [batch_size, vocab_size]
	sampled_token = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
	return sampled_token
```

*Here is the huggingface [implementation](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317).*


<br>

> A smaller `top_p` value considers only the highly probable tokens for sampling, with fewer tokens to choose from the generated text is less creative. Meanwhile a higher `top_p` value results in a more varied and creative output, with the potential risk of generating semantically incoherent text.

<br>

### **Temperature Sampling**

Temperature Sampling is a calibration method that scales down the unormalized logits using a hyperparameter T which ranges between 0 and 2, before passing them to the softmax function.

<br>

The generated predictions can sometimes be naive, in that, it could produce very peaked probability distribution that results in over-confident predictions for few tokens. To address this, we calibrate the distributation to adjust model confidence by introducing a degree of randomness using temperature. Then as usual with sampling methods, we sample at random from the resulting probability distribution:

<br>

``` python
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits)
```

<br>

- **Using a high T value `T>1`**  results in a flatter distribution, where most tokens across the range are probable. This less confident probability distribution leads to a language model output generation that is more stochasitc and creative (though this is not always case).

- **Using a low T value `T<1`** produces a peaked distribution making previously high-probability tokens even more probable. This confident probability distribution leads the language model to generate more consistent deterministic response.

<br>

> These sampling techniques can be combined together by modifying the probability distribution sequentially, starting with temperature sampling followed by the sampling methods. This ensures that the selected set of tokens have non-zero probabilities while zeroing out the probabilities of tokens that do not belong to this set. We then randomly sample the next token from the produced distribution using the `torch.multinomial` sampler, which is not to be mistaken with what `numpy.random.choice` does with uniform random sampling — used numpy as an example because the equivalent operation in Pytorch is not as straightforward — `torch.multinomial` samples from the input tensor with or without replacement (when using n>1), ensuring that the selection takes into consideration the probabilities assigned to each token, with tokens having higher probabilities being more likely to be chosen.

<br>



*Now that we understand what each strategy achieves in tuning the desired LLM response, the recommendation is as follow: you’d want to increase randomness to encourage exploration and diversity in generated text for creative tasks like storytelling (configure higher values for T | top-k | top-p). However excessive randomness can lead to meaningless outputs. Meanwhile too little to no randomness leads to a more predictable and consistent outputs, which is useful for tasks that require precise and consistent responses like machine translation (decrease T | top-k | top-p) but might cause the model to repeat itself or generate very generic responses.*
