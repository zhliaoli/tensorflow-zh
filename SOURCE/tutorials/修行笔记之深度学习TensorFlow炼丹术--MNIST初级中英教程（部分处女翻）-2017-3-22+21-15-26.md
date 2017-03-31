
# 修行笔记之深度学习TensorFlow炼丹术--MNIST初级中英教程（部分处女翻）

# 2.1MNIST For ML Beginners
# 2.1MNIST初级教程
    
     This tutorial is intended for readers who are new to both machine learning and TensorFlow. If you already know what MNIST is, and what softmax (multinomial logistic) regression is, you might prefer this faster paced tutorial. Be sure to install TensorFlow before starting either tutorial.
     这个教程的目标读者是对机器学习和TensorFlow都不太了解的新手。如果你已经了解MNIST和softmax回归(softmax regression)的相关知识，你可以阅读这个快速上手教程。在阅读任何一个教程之前请确保已经安装了TensorFlow。
     
     When one learns how to program, there's a tradition that the first thing you do is print "Hello World." Just like programming has Hello World, machine learning has MNIST.
     当我们开始学习编程的时候，第一件事往往是学习打印"Hello World"。就好比编程入门有Hello World，机器学习入门有MNIST。
     
     MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:
     MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片：

     

![avatar](http://www.tensorfly.cn/tfdoc/images/MNIST.png)

    It also includes labels for each image, telling us which digit it is. For example, the labels for the above images are 5, 0, 4, and 1.
    它也包含每一张图片对应的标签，告诉我们这个是数字几。比如，上面这四张图片的标签分别是5，0，4，1。
    
    In this tutorial, we're going to train a model to look at images and predict what digits they are. Our goal isn't to train a really elaborate model that achieves state-of-the-art performance -- although we'll give you code to do that later! -- but rather to dip a toe into using TensorFlow. As such, we're going to start with a very simple model, called a Softmax Regression.
    在此教程中，我们将训练一个机器学习模型用于预测图片里面的数字。我们的目的不是要设计一个世界一流的复杂模型 -- 尽管我们会在之后给你源代码去实现一流的预测模型--而是要介绍下如何使用TensorFlow。所以，我们这里会从一个很简单的数学模型开始，它叫做Softmax Regression。
    
    The actual code for this tutorial is very short, and all the interesting stuff happens in just three lines. However, it is very important to understand the ideas behind it: both how TensorFlow works and the core machine learning concepts. Because of this, we are going to very carefully work through the code.
    对应这个教程的实现代码很短，而且真正有意思的内容只包含在三行代码里面。但是，去理解包含在这些代码里面的设计思想是非常重要的：TensorFlow工作流程和机器学习的基本概念。因此，这个教程会很详细地介绍这些代码的实现原理。

## About this tutorial
## 关于本教程
    This tutorial is an explanation, line by line, of what is happening in the mnist_softmax.py code.
    本教程是对mnist_softmax.py中的包含的代码的逐行解释。
    
    You can use this tutorial in a few different ways, including:
    你可以使用不同的方法学习本教程，包括：

    Copy and paste each code snippet, line by line, into a Python environment as you read through the explanations of each line.
    当你读懂了每一行的注释后，你可以逐行复制粘贴每一个程序片段到Python语言的编程环境下。
    
    Run the entire mnist_softmax.py Python file either before or after reading through the explanations, and use this tutorial to understand the lines of code that aren't clear to you.
    在通读注释之前或之后，运行整个mnist_softmax.py文件，并利用本教程去弄懂那些你不清楚的代码。

    What we will accomplish in this tutorial:
    在本教程我们将完成以下步骤：

    Learn about the MNIST data and softmax regressions
    Create a function that is a model for recognizing digits, based on looking at every pixel in the image
    Use TensorFlow to train the model to recognize digits by having it "look" at thousands of examples (and run our first   TensorFlow session to do so)
    Check the model's accuracy with our test data
    学习MNIST数据集和softmax回归
    通过提取图片中的像素信息来创建一个可以识别手写数字的函数模型
    通过TensorFlow学习上千次的图片例子来训练识别手写数字的模型
    检查所得模型在测试集上的正确率


## 2.1.1The MNIST Data
##  2.1.1MNIST数据集
    The MNIST data is hosted on Yann LeCun's website. If you are copying and pasting in the code from this tutorial, start here with these two lines of code which will download and read in the data automatically:
    MNIST数据集的官网是Yann LeCun's website。你可以直接从本教程中复制粘贴到你的代码文件里面，它只有两行，但可以自动下载和安装这个数据集：
 


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
print(mnist)
```

    Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2faa6ddd30>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2faa6ddcc0>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f2faa6ddf28>)


    The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). This split is very important: it's essential in machine learning that we have separate data which we don't learn from so that we can make sure that what we've learned actually generalizes!
    下载下来的数据集可被分为三部分： 55000 行训练用点数据集（ mnist.train ）， 10000行测试数据集 ( mnist.test ) ，以及 5000 行验证数据集（ mnist.validation ）．这样的切分很重要：在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）．
    
    As mentioned earlier, every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y". Both the training set and test set contain images and their corresponding labels; for example the training images are mnist.train.images and the training labels are mnist.train.labels.
    正如前面提到的一样，每一个 MNIST 数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签．我们把这些图片设为“ x ”，把这些标签设为“ y ”．训练数据集和测试数据集都包含 x 和 y ，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels ．((注意ricequant版本为1.0.0，这里应该被标记为了xs和ys.)
    
    Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:
    每一张图片包含 28×28 像素．我们可以用一个数字数组来表示这张图片：

![avatar1](https://www.tensorflow.org/images/MNIST-Matrix.png)

    We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, as long as we're consistent between images. From this perspective, the MNIST images are just a bunch of points in a 784-dimensional vector space, with a very rich structure (warning: computationally intensive visualizations).
    我们把这个数组展开成一个向量，长度是 28×28=784 ．如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开．从这个角度来看， MNIST数据集的图片就是在 784 维向量空间里面的点 , 并且拥有比较复杂的结构 ( 注意 : 此类数据的可视化是计算密集型的 ) ．
    
    Flattening the data throws away information about the 2D structure of the image. Isn't that bad? Well, the best computer vision methods do exploit this structure, and we will in later tutorials. But the simple method we will be using here, a softmax regression (defined below), won't.
    展平图片的数字数组会丢失图片的二维结构信息．这显然是不理想的，最优秀的计算机视觉方法会挖掘并利用这些结构信息，我们会在后续教程中介绍．但是在这个教程中我们忽略这些结构，所介绍的简单数学模型， softmax 回归 (softmax regression) ，不会利用这些结构信息．
    
    The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.
    因此，在 MNIST 训练数据集中， mnist.train.images 是一个形状为 [55000, 784] 的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点．在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于 0 和 1 之间．

![2](https://www.tensorflow.org/images/mnist-train-xs.png)

    Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.
    MNIST中每张图片都有它对应的标签，这些标签为画在图片中0到9的数字.
    
    For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.
    本教程的目的，是使标签数据是 "one-hot vectors".一个 one-hot 向量除了某一位的数字是 1 以外其余各维度数字都是 0 ．所以在此教程中，数字 n 将表示成一个只有在第 n 维度（从 0 开始）数字为 1 的 10 维向量．比如，标签 0 将表示成( [1,0,0,0,0,0,0,0,0,0,0] ) ．因此， mnist.train.labels 是一个 [55000, 10] 的数字矩阵．

![3](https://www.tensorflow.org/images/mnist-train-ys.png)

    We're now ready to actually make our model!
    现在，我们准备开始真正的建模之旅！

## 2.1.2Softmax Regressions
## 2.1.2Softmax 回归
    
    We know that every image in MNIST is of a handwritten digit between zero and nine. So there are only ten possible things that a given image can be. We want to be able to look at an image and give the probabilities for it being each digit. For example, our model might look at a picture of a nine and be 80% sure it's a nine, but give a 5% chance to it being an eight (because of the top loop) and a bit of probability to all the others because it isn't 100% sure.
    我们知道 MNIST 数据集的每一张图片都表示一个 (0 到 9 的 ) 数字．所以每张图只表示十种可能之一。那么，如果模型若能看到一张图就能知道它属于各个数字的对应概率就好了。比如，我们的模型可能看到一张数字 "9" 的图片，就判断出它是数字 "9" 的概率为 80% ，而有 5% 的概率属于数字 "8" （因为 8 和 9 都有上半部分的小圆），同时给予其他数字对应的小概率（因为该图像代表它们的可能性微乎其微）．
    
    This is a classic case where a softmax regression is a natural, simple model. If you want to assign probabilities to an object being one of several different things, softmax is the thing to do, because softmax gives us a list of values between 0 and 1 that add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.
    这是能够体现 softmax 回归自然简约的一个典型案例． 因为softmax给了我们从0到加到1的一列值，softmax 模型可以用来给不同的对象分配概率．在后文，我们训练更加复杂的模型时，最后一步也往往需要用softmax 来分配概率．
    
    A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.
    softmax回归分为两步：首先对输入被分类对象属于某个类的“证据”相加求和，然后将这个“证据”的和转化为概率.
    
    To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.
    我们使用加权的方法来累积计算一张图片是否属于某类的“证据”。如果图片的像素强有力的体现该图不属于某个类，则权重为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值为正．
    
    The following diagram shows the weights one model learned for each of these classes. Red represents negative weights, while blue represents positive weights.
    下面的图片显示了一个模型学习到的图片上每个像素对于特定数字类的权值．红色代表负权值，蓝色代表正权值．

![5](https://www.tensorflow.org/images/softmax-weights.png)

    We also add some extra evidence called a bias. Basically, we want to be able to say that some things are more likely independent of the input. The result is that the evidence for a class i given an input x is:
    我们也需要引入额外的“证据”，可称之为偏置量 (bias ）.总的来说，我们希望它代表了与所输入向无关的判断证据．因此对于给定的输入图片 x 代表某数字 i 的总体证据可以表示为: 
    

 $$\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i$$

    where $W_i$ is the weights and $b_i$ is the bias for class $i$, and $j$ is an index for summing over the pixels in our input image $x$. We then convert the evidence tallies into our predicted probabilities $y$ using the "softmax" function:    
   其中，$W_i$ 代表权重，$b_i$代表第 $i$ 类的偏置量， $j$ 代表给定图片 $x$ 的像素索引用于像素求和．然后用 softmax 函数可以把这些证据转换成概率 $y$:           
                                                  $$y = \text{softmax}(\text{evidence})$$     
    Here softmax is serving as an "activation" or "link" function, shaping the output of our linear function into the form we want -- in this case, a probability distribution over 10 cases. You can think of it as converting tallies of evidence into probabilities of our input being in each class. It's defined as:   
    这里的 softmax 可以看成是一个激励（ activation ）函数或是链接（ link ）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于 10 个数字类的概率分布．因此，给定一张图片，它对于每一个数字的吻合度可以被 softmax 函数转换成为一个概率值． softmax 函数可以定义为：    
                                                 $$\text{softmax}(x) = \text{normalize}(\exp(x))$$
    If you expand that equation out, you get:    
    展开等式右边的子式，可以得到：   
                                                 $$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
    But it's often more helpful to think of softmax the first way: exponentiating its inputs and then normalizing them. The exponentiation means that one more unit of evidence increases the weight given to any hypothesis multiplicatively. And conversely, having one less unit of evidence means that a hypothesis gets a fraction of its earlier weight. No hypothesis ever has zero or negative weight. Softmax then normalizes these weights, so that they add up to one, forming a valid probability distribution. (To get more intuition about the softmax function, check out the section on it in Michael Nielsen's book, complete with an interactive visualization.)          
    但是更多的时候把 softmax 模型函数定义为第一种形式：把输入值当成幂指数求值，再正则化这些结果值．这个幂运算表示，更大的证据对应更大的假设模型（ hypothesis ）里面的乘数权重值．反之，拥有更少的证据意味着在假设模型里面拥有更小的乘数系数．假设模型里的权值不可以是 0 值或者负值． Softmax 然后会正则化这些权重值，使它们的总和等于 1 ，以此构造一个有效的概率分布．（更多的关于 Softmax 函数的信息，可以参考 Michael Nieslen 的书里面的这个部分，其中有关于 softmax 的可交互式的可视化解释．）         
    You can picture our softmax regression as looking something like the following, although with a lot more $x$s. For each output, we compute a weighted sum of the $x$s, add a bias, and then apply softmax.          
    对于 softmax 回归模型可以用下面的图解释，对于输入的 xs 加权求和，再分别加上一个偏置量，最后再输入到 softmax 函数中:

![88](https://www.tensorflow.org/images/softmax-regression-scalargraph.png)

    If we write that out as equations, we get:    
    如果把它写成一个方程，可以得到：
  

![99](https://www.tensorflow.org/images/softmax-regression-scalarequation.png)

    We can "vectorize" this procedure, turning it into a matrix multiplication and vector addition. This is helpful for computational efficiency. (It's also a useful way to think.)    
    我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加．这有助于提高计算效率（也是一种更有效的思考方式）．
    

![778](https://www.tensorflow.org/images/softmax-regression-vectorequation.png)

    More compactly, we can just write:
    更进一步，可以写成更加紧凑的方式：                 
$$y = \text{softmax}(Wx + b)$$
    Now let's turn that into something that TensorFlow can use.
    现在让我们转向TensorFlow能运用的东西：
## 2.1.3Implementing the Regression
## 2.1.3实现回归模型
    To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.     
    为了在 python 中高效的进行数值计算，我们通常会调用（如 NumPy ）外部函数库，把类似矩阵乘法这样的复杂运算使用其他外部语言实现．不幸的是，从外部计算切换回Python 的每一个操作，仍然是一个很大的开销．如果你用 GPU 来进行外部计算，这样的开销会更大．用分布式的计算方式，也会花费更多的资源用来传输数据．        
    TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. (Approaches like this can be seen in a few machine learning libraries.)    
    TensorFlow 也把复杂的计算放在 python 之外完成，但是为了避免前面说的那些开销，它做了进一步完善． TensorFlow 不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在 Python 之外运行．（这样类似的运行方式，可以在不少的机器学习库中看到．）    

    To use TensorFlow, first we need to import it：    
    使用 TensorFlow 之前，首先导入它：
    


```python
import tensorflow as tf
```

    We describe these interacting operations by manipulating symbolic variables. Let's create one:
    我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：


```python
x = tf.placeholder(tf.float32, [None, 784])
```

    x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation. We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)    
    x不是一个特定的值.它是一个占位符,当我们调用TensorFlow进行计算时才会输入数值.MNIST数据集中每张图片都被'展开'成了一个784维的向量，我们希望能输入所有这些图片的数值.我们用一个2维的浮点数张量来表示这些数，这个张量形如[None,784].(这里None表示长度可以是任意维度.)

    We also need the weights and biases for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle it: Variable. A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation. For machine learning applications, one generally has the model parameters be Variables.
    我们的模型也需要权重值和偏量值.我们可以把他们当作额外的输入，但是TensorFlow有更好的方式来表示他们：变量.变量是一个可修改的张量，它存在于TensorFlow中进行交互计算的图中.它们可以用于计算输入值，也可以在计算中被修改.对于机器学习应用而言，通常都会有变量作为模型参数.
    


```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

    We create these Variables by giving tf.Variable the initial value of the Variable: in this case, we initialize both W and b as tensors full of zeros. Since we are going to learn W and b, it doesn't matter very much what they initially are.
    我们赋予 tf.Variable 不同的初值来创建不同的 Variable ：在这里，我们都用全为零的张量来初始化 W 和 b ．因为我们要学习 W 和 b 的值，它们的初值可以随意设置．

    Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes. b has a shape of [10] so we can add it to the output.
    注意， W 的维度是 [784，10] ，因为我们想要用 784 维的图片向量乘以它以得到一个10 维的证据值向量，每一位对应不同数字类． b 的形状是 [10] ，所以我们可以直接把它加到输出上面．

    We can now implement our model. It only takes one line to define it!
    现在可以实现我们的模型了.定义它只需要一行代码！


```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

   First, we multiply $x$ by $W$ with the expression $tf.matmul(x, W)$. This is flipped from when we multiplied them in our equation, where we had $Wx$, as a small trick to deal with $x$ being a 2D tensor with multiple inputs. We then add $b$, and finally apply $tf.nn.softmax$.     
   首先，我们通过表达式$tf.matmul(x, W)$来表示$x$乘以$W$.对应之前等式里的$Wx$,这里 $x$ 是一个 2 维张量拥有多个输入．然后再加上 $b$ ，把和输入到 $tf.nn.softmax$ 函数里面．

    That's it. It only took us one line to define our model, after a couple short lines of setup. That isn't because TensorFlow is designed to make a softmax regression particularly easy: it's just a very flexible way to describe many kinds of numerical computations, from machine learning models to physics simulations. And once defined, our model can be run on different devices: your computer's CPU, GPUs, and even phones!
    至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型． TensorFlow 不仅仅可以使 softmax 回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型．一旦被定义好之后，我们的模型就可以在不同的设备上运行：计算机的 CPU ， GPU ，甚至是手机！

    

## 2.1.4Training
## 2.1.4训练模型
    In order to train our model, we need to define what it means for the model to be good. Well, actually, in machine learning we typically define what it means for a model to be bad. We call this the cost, or the loss, and it represents how far off our model is from our desired outcome. We try to minimize that error, and the smaller the error margin, the better our model is.
    为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的．其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（ cost ）或损失（ loss ），它表示离我们期望达到的目标有多远.我们尽量最小化这个错误，样本外错误越小，我们模型的效果越好．
    One very common, very nice function to determine the loss of a model is called "cross-entropy." Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning. It's defined as:
    一个非常常见的，非常漂亮的成本函数是“交叉熵” (cross-entropy) ．交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段．它的定义如下：
$$H_{y'}(y) = -\sum_i y'_i \log(y_i)$$   
    Where $y$ is our predicted probability distribution, and $y′$ is the true distribution (the one-hot vector with the digit labels). In some rough sense, the cross-entropy is measuring how inefficient our predictions are for describing the truth. Going into more detail about cross-entropy is beyond the scope of this tutorial, but it's well worth understanding.    
    y 是我们预测的概率分布 ,y ′ 是实际的分布（我们输入的 one-hot vector) ．比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性．更详细的关于交叉熵的解释超出本教程的范畴，但是你很有必要好好理解它．
    

    To implement cross-entropy we need to first add a new placeholder to input the correct answers:
    为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

   Then we can implement the cross-entropy function,$-\sum y'\log(y)$     
   然后我们可以用$-\sum y'\log(y)$计算交叉熵 :


```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

```

    First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.   
    首先，用 tf.log 计算 y 的每个元素的对数．接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘．由于参数reduction_indices=[1]，我们用 tf.reduce_sum 计算张量的所有元素的总和．最后，用tf.reduce_mean计算所有分支实例的平均值.

       Note that in the source code, we don't use this formulation, because it is numerically unstable. Instead, we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b), because this more numerically stable function internally computes the softmax activation. In your code, consider using tf.nn.softmax_cross_entropy_with_logits instead.
    （没翻译）

    Now that we know what we want our model to do, it's very easy to have TensorFlow train it to do so. Because TensorFlow knows the entire graph of your computations, it can automatically use the backpropagation algorithm to efficiently determine how your variables affect the loss you ask it to minimize. Then it can apply your choice of optimization algorithm to modify the variables and reduce the loss.
    现在我们知道我们需要我们的模型做什么啦，用 TensorFlow 来训练它是非常容易的．因为 TensorFlow 拥有一张描述你各个计算单元的图，它可以自动地使用反向传播算法 (backpropagation algorithm) 来有效地确定你的变量是如何影响你想要最小化的那个成本值的．然后， TensorFlow 会用你选择的优化算法来不断地修改变量以降低损失．


```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

    In this case, we ask TensorFlow to minimize $cross_entropy$ using the gradient descent algorithm with a learning rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost. But TensorFlow also provides many other optimization algorithms: using one is as simple as tweaking one line.
     在这里，我们要求 TensorFlow 用梯度下降算法（ gradient descent algorithm ）以 0.5的学习速率最小化交叉熵．梯度下降算法（ gradient descent algorithm ）是一个简单的学习过程， TensorFlow 只需将每个变量一点点地往使成本不断降低的方向移动．当然TensorFlow 也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法．
   
   What TensorFlow actually does here, behind the scenes, is to add new operations to your graph which implement backpropagation and gradient descent. Then it gives you back a single operation which, when run, does a step of gradient descent training, slightly tweaking your variables to reduce the loss.   
   TensorFlow 在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法．然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少损失．

    We can now launch the model in an $InteractiveSession$:
    现在我们可以在$InteractiveSession$启动模型：
    


```python
sess = tf.InteractiveSession()
```

    We first have to create an operation to initialize the variables we created:
    我们首先需要创建一个操作去初始化我们创建的变量：


```python
tf.global_variables_initializer().run()
```

    Let's train -- we'll run the training step 1000 times!
    然后开始训练模型，这里我们让模型循环训练 1000 次！


```python
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

    Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the $placeholders$.
    该循环的每个步骤中，我们都会随机抓取训练数据中的 100 个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行 train_step ．

    Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
    使用一小部分的随机数据来进行训练被称为随机训练 (stochastic training)--- 在这里更确切的说是随机梯度下降训练．理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销．所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性.

## 2.1.5Evaluating Our Model
## 2.1.5评估我们的模型
    How well does our model do?
    我们的模型性能如何呢？

    Well, first let's figure out where we predicted the correct label. $tf.argmax$ is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, $tf.argmax(y,1)$ is the label our model thinks is most likely for each input, while $tf.argmax(y_,1)$ is the correct label. We can use $tf.equal$ to check if our prediction matches the truth.
    首先让我们找出那些预测正确的标签． $tf.argmax()$ 是一个非常有用的函数，它能给你在一个张量里沿着某条轴的最高条目的索引值．比如， $tf.argmax(y,1) $是模型认为每个输入最有可能对应的那些标签，而$ tf.argmax(y_,1)$ 代表正确的标签．我们可以用$ tf.equal$
来检测我们的预测是否真实标签匹配.


```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

    That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
    这行代码会给我们一组布尔值．为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值．例如， [True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75 .


```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

    Finally, we ask for our accuracy on our test data.
    最后，我们计算所学习到的模型在测试数据集上面的正确率．


```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    0.9192


    This should be about 91%.
    最终结果值应该大约是 91% ．

    Is that good? Well, not really. In fact, it's pretty bad. This is because we're using a very simple model. With some small changes, we can get to 97%. The best models can get to over 99.7% accuracy! (For more information, have a look at this list of results.)
    这个结果好吗？嗯，并不太好．事实上，这个结果是很差的．这是因为我们仅仅使用了一个非常简单的模型．不过，做一些小小的改进，我们就可以得到 97% 的正确率．最好的模型甚至可以获得超过 99.7% 的准确率！（想了解更多信息，请参考这个结果对比列表． 

    What matters is that we learned from this model. Still, if you're feeling a bit down about these results, check out the next tutorial where we do a lot better, and learn how to build more sophisticated models using TensorFlow!
    比结果更重要的是，我们从这个模型中学习到的设计思想．不过，如果你仍然对这里的结果有点失望，可以查看下个教程，在那里你将学到如何用 FensorFlow 构建更加复杂的模型以获得更好的性能！


```python
mnist.test.images?
```


```python
mnist.test.labels?
```


```python

```
