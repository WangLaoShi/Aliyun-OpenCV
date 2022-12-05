### 数据介绍

数据地址：https://github.com/hromi/SMILEsmileD

数据包含13165张灰度图片，每张图片的尺寸是64*64。这个数据集并不算平衡，13165张图片中，有9475张图片不是笑脸图片，有3690张图片是笑脸图片。数据差异很大。

### 数据预处理

首先导入相应的包：

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import os

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
```


```python
dataset_dir = os.path.abspath(r"./SMILEs/") #smile数据集路径
model_dir = os.path.abspath(r"./model/lenet.hdf5")    #训练模型保存路径

data = []
labels = []
```

```python
for imagePath in sorted(list(paths.list_images(dataset_dir))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # 转换成灰度图像
    image = imutils.resize(image, width = 28)  #将图像尺寸改成28*28
    image = img_to_array(image)   #使用Keras的img_to_array转换成浮点型和（28*28*1），便于接下来神经网络学习
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"  #如果label字符串里面有positive就重命名为smiling
    labels.append(label)
```

```python
# 将data和labels都转换为numpy类型
data = np.array(data, dtype= "float") / 255.0 #将像素转换到[0, 1]范围之内
labels = np.array(labels)

# 对label进行one-hot编码
le = LabelEncoder().fit(labels)   # LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 

# transform用来标准化，将labels中'not_smiling'和‘smiling’的数据转换成0和1的形式
labels = np_utils.to_categorical(le.transform(labels), 2)  # 2是num_class表示输出的是2列数据的意思
```
下面需要解决一下样本不平衡问题。

数据集里面有9475个笑脸样本，和3690个非笑脸样本。下面的代码中classTotals就是按列加和labels的one-hot编码，所以结果是[9475, 3690] 我们要解决数据不平衡问题可以使用classWeight权重，相比于笑脸，我们给非笑脸以2.56倍的权重。损失函数权重计算的时候对非笑脸进行相应扩大，以此来解决数据不平衡问题。


```python
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
```
stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下：

training: 75个数据，其中60个属于A类，15个属于B类。

testing: 25个数据，其中20个属于A类，5个属于B类。

用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify

```python
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, 
                                                 stratify = labels, random_state = 42)
```

### 使用LeNet实现笑脸检测分类

下面是模型实现部分：

```python
model = Sequential()

# first set of CONV => RELU => POOL layers
model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=20, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

# second set of CONV => RELU => POOL layers
model.add(Conv2D(kernel_size=(5, 5), filters=50,  activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500, activation='relu'))

model.add(Dense(2, activation='softmax'))
```

```python
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

H = model.fit(trainX, trainY, validation_data = (testX, testY),
             class_weight = classWeight, batch_size = 64, epochs = 15, verbose = 1)  #verbose = 1显示进度条
```
keras没有直接可以统计recall和f1值的办法。可以用sklearn。 但是sklearn没有办法直接处理Keras的数据，所以要经过一些处理。Keras计算需要二维数组，但classification_report可以处理的是一维数列，所以这里使用argmax按行返回二维数组最大索引，这样也算是一种0-1标签的划分了。

```python
predictions = model.predict(testX, batch_size = 64)


print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                           target_names = le.classes_))  # le.classes是['not_smiling', 'smiling']组成的数组

model.save(model_dir)
```
输出结果：
```
             precision    recall  f1-score   support

not_smiling       0.95      0.91      0.93      1895
    smiling       0.79      0.87      0.83       738

avg / total       0.90      0.90      0.90      2633
```

```python
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label = "acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")le
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

![image](https://github.com/Einstellung/DataScience_learning/blob/master/DataScience/images/SmileDetection/1.png?raw=true)


### 人脸检测实现

这里使用OpenCV的Haar特征和级联分类器来实现实时人脸检测，关于Haar特征和级联分类器的理论知识，[可以看这里](https://blog.csdn.net/Einstellung/article/details/89352853)

我们在代码中使用了OpenCV这个工具来具体实现，在OpenCV中，相应算法都已经做好了封装，直接调用就可以了。值得一提的是，人脸检测的模型已经提前训练好了，这里我们直接调用模型就可以在，是一个XML格式的文件“haarcascade_frontalface_default.xml”，一般在opencv-3.4\opencv\sources\data\haarcascades路径下可以找到。下面是具体代码实现：


```python
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required= True,
               help= "path to where the face cascade resides")
ap.add_argument("-m", "--model", required= True,
               help= "path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())


detector = cv2.CascadeClassifier(args["cascade"])  #从对应路径中加载人脸检测级联分类器
model = load_model(args["model"])

# 对是从相机中检测人脸还是从视频中检测人脸做判断
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:

# grabbed和frame是read的两个返回值，grabbed是布尔类型的返回值，如果读取帧是正确的返回True，当文件读到结尾的时候返回False
# frame是每一帧的图像，是一个三维矩阵
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

        
    frame = imutils.resize(frame, width = 300)   #把图像宽度重新指定为300像素
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #因为模型训练是对灰度图像处理，所以这里要转换成灰度图像
    frameClone = frame.copy()  #重新克隆frame，用于接下来绘制边界框
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)
        
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"
        
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                     (0, 0, 255), 2)
        
        
    cv2.imshow("Face", frameClone)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()
```
#### detector.detectMultiScale

这里，对`detector.detectMultiScale`做一点说明：

为了检测到不同大小的目标，一般有两种做法：逐步缩小图像；或者，逐步放大检测窗口。缩小图像就是把图像长宽同时按照一定比例（默认1.1 or 1.2）逐步缩小，然后检测；放大检测窗口是把检测窗口长宽按照一定比例逐步放大，这时位于检测窗口内的特征也会对应放大，然后检测。在默认的情况下，OpenCV是采取逐步缩小的情况，如下图所示，最先检测的图片是底部那张大图。

![image](https://img-blog.csdn.net/20180824100209189?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3Rhbm14MjE5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后，对应每张图，级联分类器的大小固定的检测窗口器开始遍历图像，以便在图像找到位置不同的目标。对照程序来看，这个固定的大小就是上图的红色框。


```python

void CascadeClassifier::detectMultiScale( InputArray image,
                      CV_OUT std::vector<Rect>& objects,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      Size minSize,
                      Size maxSize )
```
参数1：image–待检测图片，一般为灰度图像以加快检测速度；

参数2：objects–被检测物体的矩形框向量组；为输出量，如某特征检测矩阵Mat

参数3：scaleFactor–表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%

参数4：minNeighbors–表示构成检测目标的相邻矩形的最小个数(默认为3个)。 如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。 如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框， 这种设定值一般用在用户自定义对检测结果的组合程序上；

参数5：flags=0：可以取如下这些值： 
CASCADE_DO_CANNY_PRUNING=1, 利用canny边缘检测来排除一些边缘很少或者很多的图像区域 
CASCADE_SCALE_IMAGE=2, 正常比例检测 
CASCADE_FIND_BIGGEST_OBJECT=4, 只检测最大的物体 
CASCADE_DO_ROUGH_SEARCH=8 初略的检测 
6. minObjectSize maxObjectSize：匹配物体的大小范围

参数6、7：minSize和maxSize用来限制得到的目标区域的范围。也就是我本次训练得到实际项目尺寸大小 函数介绍： detectMultiscale函数为多尺度多目标检测： 多尺度：通常搜索目标的模板尺寸大小是固定的，但是不同图片大小不同，所以目标对象的大小也是不定的，所以多尺度即不断缩放图片大小（缩放到与模板匹配），通过模板滑动窗函数搜索匹配；同一副图片可能在不同尺度下都得到匹配值，所以多尺度检测函数detectMultiscale是多尺度合并的结果。 多目标：通过检测符合模板匹配对象，可得到多个目标，均输出到objects向量里面。

minNeighbors=3：匹配成功所需要的周围矩形框的数目，每一个特征匹配到的区域都是一个矩形框，只有多个矩形框同时存在的时候，才认为是匹配成功，比如人脸，这个默认值是3。


因为代码中使用了argparse，所以可以通过命令来指定，如果是想用webcam（PC自带摄像头）的话，可以输入：

```python
python detect_smile.py --cascade haarcascade_frontalface_default.xml 
                       --model output/lenet.hdf5
```
如果是用视频的话，可以输入如下命令：

```python
python detect_smile.py --cascade haarcascade_frontalface_default.xml 
                       --model output/lenet.hdf5
                       --video path/to/your/video.mov
```

