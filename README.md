# FER
基于CNN的FER
## 1. 数据集选择  
FER2013包含七种情感，分别为愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性，共35887张灰度图像。数据最初通过网络收集，包含不同年龄、种族和性别的面部图像，增加了数据的多样性，但同时也带来了噪声和标注错误。由于图像分辨率较低，表情的细微差异在视觉上不太明显。此外，不同个体的表情差异较大，加之数据集中各情感类别的分布不均匀，这为模型的训练带来了挑战。  
数据集的下载地址为：<https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data>  
## 2. 模型选择  
CNN具有局部特征不变性、权值共享等特点。在处理图像任务上表现优异，能够有效地提取出图像的多层次特征，具有高效的空间不变性和较强的泛化能力，是目前图像处理任务的主流模型。  

算法：用于面部表情识别的卷积神经网络（CNN）  
输入：大小为 48x48 的灰度图像  
输出：7 类情感类别中的一种  

步骤：  

Ⅰ初始化一个顺序模型。  

Ⅱ添加卷积层：  

&emsp;2x卷积层：64 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
&emsp;2x卷积层：128 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
&emsp;2x卷积层：256 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
Ⅲ添加全连接层：  

&emsp;扁平化层。  
&emsp;全连接层：128 个单元，激活函数 ELU，BatchNormalization。  
&emsp;输出层：7 个单元，激活函数 softmax。  

Ⅳ编译模型：使用 Adam 优化器（学习率 $\alpha$=0.001），损失函数为 categorical_crossentropy，评估指标为准确率accuracy。  

Ⅴ训练模型：训练 100 个周期，在测试集上进行评估。  

输出：返回训练好的模型及其性能指标。  

## 3. 训练和优化
