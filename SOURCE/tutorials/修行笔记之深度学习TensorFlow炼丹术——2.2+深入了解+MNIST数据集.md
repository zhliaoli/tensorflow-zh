
# 2.2  Deep MNIST for Experts  
# 2.2 深入了解 MNIST数据集
    TensorFlow is a powerful library for doing large-scale numerical computation. One of the tasks at which it excels is implementing and training deep neural networks. In this tutorial we will learn the basic building blocks of a TensorFlow model while constructing a deep convolutional MNIST classifier.
    TensorFlow 是一个善于大规模数值计算的强大库件。它的一个强项就是训练并实现深度神经网络 (deep neural networks) 。在本小节中，我们将会学习 TensorFlow 模型构建的基本方法，并以此构建一个深度卷积 MNIST 分类器。

    This introduction assumes familiarity with neural networks and the MNIST dataset. If you don't have a background with them, check out the introduction for beginners. Be sure to install TensorFlow before starting.
    本教程假设您已经熟悉神经网络和 MNIST 数据集。如果你尚未了解，请查看新手指南。再开始学习前请确保您已安装 TensorFlow 。

## About this tutorial
## 关于本教程
    The first part of this tutorial explains what is happening in the mnist_softmax.py code, which is a basic implementation of a Tensorflow model. The second part shows some ways to improve the accuracy.
    本教程第一部分解释了mnist_softmax.py中的包含的代码的如何运行的,这是一个TensorFlow模型的基本实例。第二部分写了几种提高准确率的方法.
    
    You can copy and paste each code snippet from this tutorial into a Python environment, or you can choose to just read through the code.
    你可以逐行复制粘贴每一个程序片段到Python语言的编程环境下,你也可以选择只是读下代码。
    
    What we will accomplish in this tutorial:
    在本教程我们将完成以下步骤：

    Create a softmax regression function that is a model for recognizing MNIST digits, based on looking at every pixel in the image
    Use Tensorflow to train the model to recognize digits by having it "look" at thousands of examples (and run our first Tensorflow session to do so)
    Check the model's accuracy with our test data
    Build, train, and test a multilayer convolutional neural network to improve the results
    通过提取图片中的像素信息来创建一个可以识别MNIST手写数字的softmax回归函数模型
    通过TensorFlow学习上千次的图片例子来训练识别手写数字的模型
    检查所得模型在测试集上的正确率
    创建，训练和测试一个多重卷积神经网络来提高结果
    
## Setup    
    Before we create our model, we will first load the MNIST dataset, and start a TensorFlow session.
    在创建模型前，我们首先要下载MNIST数据集，然后加载一个TensorFlow会话.
    
## 2.2.1The MNIST Data
##  2.2.1MNIST数据集
    The MNIST data is hosted on Yann LeCun's website. If you are copying and pasting in the code from this tutorial, start here with these two lines of code which will download and read in the data automatically:
    MNIST数据集的官网是Yann LeCun's website。你可以直接从本教程中复制粘贴到你的代码文件里面，它只有两行，但可以自动下载和安装这个数据集：
 


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz


    Here mnist is a lightweight class which stores the training, validation, and testing sets as NumPy arrays. It also provides a function for iterating through data minibatches, which we will use below.
    此处的 mnist 是一个以 NumPy 数组形式存储训练、验证和测试数据的轻量级类。我们将在之后使用到它提供的一个函数功能，用于迭代按批处理数据。
### Start TensorFlow InteractiveSession
### 开始 TensorFlow 交互会话
    TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.
    Tensorflow 基于一个高效的 C++ 后台模块进行运算。与这个后台模块的连接叫做会话 (session) 。 TensorFlow 编程的常规流程是先创建一个图，然后在 session 中加载它。

    Here we instead use the convenient InteractiveSession class, which makes TensorFlow more flexible about how you structure your code. It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.
    这里，我们使用更加方便的交互会话 (InteractiveSession) 类，它可以让您更加灵活地构建代码。交互会话能让你在运行图的时候，插入一些构建计算图的操作。这能给使用交互式文本 shell 如 iPython 带来便利。如果你没有使用 InteractiveSession 的话，你需要在开始 session 和加载图之前，构建整个计算图。


```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

### Computation Graph
### 计算图
    To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.
    为了高效地在 Python 里进行数值计算，我们一般会使用像 NumPy 这样用其他语言编写的库件，在 Python 外用其它执行效率高的语言完成这些高运算开销操作（如矩阵运算）。但是，每一步操作依然会需要切换回 Python 带来很大开销。特别的，这种开销会在 GPU 运算或是分布式集群运算这类高数据传输需求的运算形式上非常高昂。
    
    TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in Theano or Torch.
    TensorFlow 将高运算量计算放在 Python 外进行，同时更进一步设法避免上述的额外运算开销。不同于在 Python 中独立运行运算开销昂贵的操作， TensorFlow 让我们可以独立于 Python 以外以图的形式描述交互式操作。这与 Theano 、 Torch 的做法很相似。
    
    The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run. See the Computation Graph section of Getting Started With TensorFlow for more detail.
    因此，这里 Python 代码的角色是构建其外部将运行的计算图，并决定计算图的哪一部分将被运行。更多的细节请参阅开始运行tensorFlow章节中计算图的部分。

## 2.2.2Build a Softmax Regression Model
## 2.2.2建立一个Softman回归模型

    In this section we will build a softmax regression model with a single linear layer. In the next section, we will extend this to the case of softmax regression with a multilayer convolutional network.
    在这小节里，我们将会构建一个包含单个线性隐层的 softmax 回归模型。我们将在下一小结把它扩展成多层卷积网络 softmax 回归模型。
    
###  Placeholders
###  占位符

    We start building the computation graph by creating nodes for the input images and target output classes.
    我们先从创建输入图像和输出类别的节点来创建计算图。


```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

    Here x and y_ aren't specific values. Rather, they are each a placeholder -- a value that we'll input when we ask TensorFlow to run a computation.
    这里的 x 和 y 并不代表具体值，他们是一个占位符 ( placeholder ) — 当 TensorFlow 运行时需要赋值的变量。
    
    The input images x will consist of a 2d tensor of floating point numbers. Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image, and None indicates that the first dimension, corresponding to the batch size, can be of any size. The target output classes y_ will also consist of a 2d tensor, where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.
    输入图片 x 是由浮点数值组成的 2 维张量 (tensor) 。这里，我们定义它为 [None, 784]的 shape ，其中 784 是单张展开的 MNIST 图片的维度数。 None 对应 shape 的第一个维度，代表了这批输入图像的数量，可能是任意值。目标输出类 y_ 也是一个 2 维张量，其中每一行为一个 10 维向量代表对应 MNIST 图片的所属数字的类别。

    The shape argument to placeholder is optional, but it allows TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes.
    虽然 placeholder 的 shape 参数是可选的，但有了它， TensorFlow 能够自动捕捉因数据维度不一致导致的错误。

### Variables
### 变量
    
    We now define the weights W and biases b for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle them: Variable. A Variable is a value that lives in TensorFlow's computation graph. It can be used and even modified by the computation. In machine learning applications, one generally has the model parameters be Variables.
    我们现在为模型定义权重 W 和偏置 b 。它们可以被视作是额外的输入量，但是 TensorFlow 有一个更好的方式来处理： Variable 。一个 Variable 代表着在 TensorFlow 计算图中的一个值，它是能在计算过程中被读取和修改的。在机器学习的应用过程中，模型参数一般用 Variable 来表示。


```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

    We pass the initial value for each parameter in the call to tf.Variable. In this case, we initialize both W and b as tensors full of zeros. W is a 784x10 matrix (because we have 784 input features and 10 outputs) and b is a 10-dimensional vector (because we have 10 classes).
    我们在调用 tf.Variable 的时候传入初始值。在这个例子里，我们把 W 和 b 都初始化为零向量。 W 是一个 784×10 的矩阵（因为我们有 784 个特征和 10 个输出值）。 b 是一个 10维的向量（因为我们有 10 个分类）。

    Before Variables can be used within a session, they must be initialized using that session. This step takes the initial values (in this case tensors full of zeros) that have already been specified, and assigns them to each Variable. This can be done for all Variables at once:
    变量需要在 session 之前初始化，才能在 session 中使用。初始化需要初始值（本例当中是全为零）传入并赋值给每一个 Variable 。这个操作可以一次性完成。


```python
sess.run(tf.global_variables_initializer())
```

### Predicted Class and Loss Function
### 预测分类和损失函数

    We can now implement our regression model. It only takes one line! We multiply the vectorized input images x by the weight matrix W, add the bias b.
    现在我们可以实现我们的 regression 模型了。这只需要一行！我们把图片 x 和权重矩阵 W 相乘，加上偏置 b。


```python
y = tf.matmul(x,W) + b
```

    We can specify a loss function just as easily. Loss indicates how bad the model's prediction was on a single example; we try to minimize that while training across all the examples. Here, our loss function is the cross-entropy between the target and the softmax activation function applied to the model's prediction. As in the beginners tutorial, we use the stable formulation:
    我们指定一个损失函数同样简单。损失函数表明了模型在单一例子上的槽糕程度。当我们训练整个样本时，我们试图使它的值最小。我们这里的损失函数用目标分类和使用softmax激活函数预测分类之间的交叉熵。和初级教程中的一样，我们用固定的公式：


```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

    Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.
    注意 tf.nn.softmax_cross_entropy_with_logits在内部使用softmax函数计算非标准化的模型预测值，然后把各分类得到的值相加，最后用tf.reduce_mean求得这些和求平均值。

## 2.2.3 Train the Model
## 2.2.3 训练模型

    Now that we have defined our model and training loss function, it is straightforward to train using TensorFlow. Because TensorFlow knows the entire computation graph, it can use automatic differentiation to find the gradients of the loss with respect to each of the variables. TensorFlow has a variety of built-in optimization algorithms. For this example, we will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.
    我们已经定义好了模型和训练的时候用的损失函数，接下来使用 TensorFlow 来训练。因为 TensorFlow 知道整个计算图，它会用自动微分法来找到损失函数对于各个变量的梯度。 TensorFlow 有大量内置优化算法，这个例子中，我们用快速梯度下降法让交叉熵下降，步长为 0.5 。


```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

    What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.
    这一行代码实际上是用来往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值。

    The returned operation train_step, when run, will apply the gradient descent updates to the parameters. Training the model can therefore be accomplished by repeatedly running train_step.
    train_step 这个操作，用梯度下降来更新权值。因此，整个模型的训练可以通过反复地运行 train_step 来完成。


```python
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

    We load 100 training examples in each training iteration. We then run the train_step operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. Note that you can replace any tensor in your computation graph using feed_dict -- it's not restricted to just placeholders.
    每一步迭代，我们都会加载 50 个训练样本，然后执行一次 train_step，使用feed_dict，用训练数据替换 placeholder 向量 x 和 y_ 。注意，在计算图中，你可以用 feed_dict 来替代任何张量，并不仅限于替换 placeholder 。

### Evaluate the Model
### 评估模型

    How well did our model do?
    我们的模型效果怎样？

    First we'll figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the true label. We can use tf.equal to check if our prediction matches the truth.
    首先，要先知道我们哪些 label 是预测正确了。 tf.argmax 是一个非常有用的函数。它会返回一个张量某个维度中的最大值的索引。例如， tf.argmax(y,1) 表示我们模型对每个输入的最大概率分类的分类值。而 tf.argmax(y_,1) 表示真实分类值。我们可以用 tf.equal 来判断我们的预测是否与真实分类一致。


```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

    That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
    这里返回一个布尔数组。为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对、错，然后取平均值。例如： [True, False, True, True] 变为 [1,0,1,1] ，计算出平均值为 0.75 。


```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

    Finally, we can evaluate our accuracy on the test data. This should be about 92% correct.
    最后，我们可以计算出在测试数据上的准确率，大概是 92% 。


```python
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    0.92


## 2.2.4 Build a Multilayer Convolutional Network
## 2.2.4 构建多层卷积网络模型
    Getting 92% accuracy on MNIST is bad. It's almost embarrassingly bad. In this section, we'll fix that, jumping from a very simple model to something moderately sophisticated: a small convolutional neural network. This will get us to around 99.2% accuracy -- not state of the art, but respectable.
    在 MNIST 上只有 91% 正确率，实在太糟糕。在这个小节里，我们用一个稍微复杂的模型：卷积神经网络来改善效果。这会达到大概 99.2% 的准确率。虽然不是最高，但是还是比较让人满意。

### Weight Initialization
### 权重初始化

    To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.
    在创建模型之前，我们先来创建权重和偏置。一般来说，初始化时应加入轻微噪声，来打破对称性，防止零梯度的问题。因为我们用的是 ReLU ，所以用稍大于 0 的值来初始化偏置能够避免节点输出恒为 0 的问题（ dead neurons ）。为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。


```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

### Convolution and Pooling
###  卷积和池化

    TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations into functions.
    TensorFlow 在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用 vanilla 版本。我们的卷积使用 1 步长（ stride size ），0 边距（ padding size ）的模板，保证输出和输入是同一个大小。我们的池化用简单传统的 2×2 大小的模板做 max pooling 。为了代码更简洁，我们把这部分抽象成一个函数


```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

### First Convolutional Layer
### 第一层卷积

    We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
    现在我们可以开始实现第一层了。它由一个卷积接一个 max pooling 完成。卷积在每个 5×5 的 patch 中算出 32 个特征。权重是一个 [5, 5, 1, 32] 的张量，前两个维度是patch 的大小，接着是输入的通道数目，最后是输出的通道数目。输出对应一个同样大小的偏置向量。


```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

    To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.
    为了用这一层，我们把 x 变成一个 4d 向量，第 2 、 3 维对应图片的宽高，最后一维代表颜色通道。


```python
x_image = tf.reshape(x, [-1,28,28,1])
```

    We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. The max_pool_2x2 method will reduce the image size to 14x14.
    我们把 x_image 和权值向量进行卷积相乘，加上偏置，使用 ReLU 激活函数，最后 maxpooling 。max_pool_2x2方法将使图片的维度变为 14x14.


```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### Second Convolutional Layer
### 第二卷积层

    In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.
    为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个 5x5的patch 会得到 64 个特征。


```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

###  Densely Connected Layer
###  密集连接层

    Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
    现在，图片降维到 7×7 ，我们加入一个有 1024 个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，使用 ReLU 激活。


```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

### Dropout

    To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.
    为了减少过拟合，我们在输出层之前加入 dropout 。我们用一个 placeholder 来代表一个神经元在 dropout 中被保留的概率。这样我们可以在训练过程中启用 dropout ，在测试过程中关闭 dropout 。 TensorFlow 的 tf.nn.dropout 操作会自动处理神经元输出值的scale 。所以用 dropout 的时候可以不用考虑 scale 。


```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### Readout Layer
### 输出层

    Finally, we add a layer, just like for the one layer softmax regression above.
    最后，我们添加一个 softmax 层，就像前面的单层 softmax regression 一样


```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

### Train and Evaluate the Model
### 训练和评估模型

    How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above.
    这次效果又有多好呢？我们用前面几乎一样的代码来测测看。为了训练和评估它，我们将使用与上述简单的一层SoftMax网络几乎相同的代码。

    The differences are that:
    差异在于：

    We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
    We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
    We will add logging to every 100th iteration in the training process.
    我们会用更加复杂的 ADAM 优化器来做梯度最速下降
    我们在 feed_dict 中加入额外的参数 keep_prob 来控制 dropout 比例
    我们每 100 次迭代输出一次日志
    Feel free to go ahead and run this code, but it does 20,000 training iterations and may take a while (possibly up to half an hour), depending on your processor.
    随意运行这段代码，但是它会执行20,000次训练迭代，并且可能需要一段时间（可能长达半小时），这取决于您的处理器。


```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

    step 0, training accuracy 0.08
    step 100, training accuracy 0.88
    step 200, training accuracy 0.96
    step 300, training accuracy 0.88
    step 400, training accuracy 0.94
    step 500, training accuracy 0.98
    step 600, training accuracy 0.92
    step 700, training accuracy 0.94
    step 800, training accuracy 0.94


    The final test set accuracy after running this code should be approximately 99.2%.
    以上代码，在最终测试集上的准确率大概是 99.2% 。

    We have learned how to quickly and easily build, train, and evaluate a fairly sophisticated deep learning model using TensorFlow.
    目前为止，我们已经学会了用 TensorFlow 来快速和简易地搭建、训练和评估一个复杂一点儿的深度学习模型。

   


```python

```


```python

```


```python

```
