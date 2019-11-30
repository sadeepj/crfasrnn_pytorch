# CRF-RNN for Semantic Image Segmentation - PyTorch version
![sample](sample.png)

<b>Live demo:</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [http://crfasrnn.torr.vision](http://crfasrnn.torr.vision) <br/>
<b>Caffe version:</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[http://github.com/torrvision/crfasrnn](http://github.com/torrvision/crfasrnn)<br/>
<b>Tensorflow/Keras version:</b> [http://github.com/sadeepj/crfasrnn_keras](http://github.com/sadeepj/crfasrnn_keras)<br/>

This repository contains the official PyTorch implementation of the "CRF-RNN" semantic image segmentation method, published in the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf). The [online demo](http://crfasrnn.torr.vision) of this project won the Best Demo Prize at ICCV 2015. Results of this PyTorch code are identical to that of the Caffe and Tensorflow/Keras based versions above.

If you use this code/model for your research, please cite the following paper:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and
    Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```

## Installation Guide

_Note_: If you are using a Python virtualenv, make sure it is activated before running each command in this guide.

### Step 1: Clone the repository
```
$ git clone https://github.com/sadeepj/crfasrnn_pytorch.git
```
The root directory of the clone will be referred to as `crfasrnn_pytorch` hereafter.

### Step 2: Install dependencies


Use the `requirements.txt` file in this repository to install all the dependencies via `pip`:
```
$ cd crfasrnn_pytorch
$ pip install -r requirements.txt
```

After installing the dependencies, run the following commands to make sure they are properly installed:
```
$ python
>>> import torch 
```
You should not see any errors while importing `torch` above.

### Step 3: Build CRF-RNN custom op

Run `setup.py` inside the `crfasrnn_pytorch/crfasrnn` directory:
```
$ cd crfasrnn_pytorch/crfasrnn
$ python setup.py install 
``` 
Note that the `python` command in the console should refer to the Python interpreter associated with your PyTorch installation. 

### Step 4: Download the pre-trained model weights

Download the model weights from [here](https://github.com/sadeepj/crfasrnn_pytorch/releases/download/0.0.1/crfasrnn_weights.pth) and place it in the `crfasrnn_pytorch` directory with the file name `crfasrnn_weights.pth`.

### Step 5: Run the demo
```
$ cd crfasrnn_pytorch
$ python run_demo.py
```
If all goes well, you will see the segmentation results in a file named "labels.png".

## Contributors
* Sadeep Jayasumana ([sadeepj](https://github.com/sadeepj))
* Harsha Ranasinghe ([HarshaPrabhath](https://github.com/HarshaPrabhath))

