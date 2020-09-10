# TSNE

__Visulize the Intermidiate features of CNN models using TSNE__

In this repo, we use ResNet18 model to classify CIFAR-10 dataset. Then we apply the PCA+TSNE to visualize the features of last FC layer.

## Demo of testing data
![Img](https://raw.githubusercontent.com/13952522076/TSNE/master/images/result.gif)

## Get started
Clone this repo and run the code
```bash
$ git clone https://github.com/13952522076/TSNE.git
$ cd TSNE
$ CUDA_VISIBLE_DEVICES=0 python3 cifar.py  
```
Optional paramters:
'--number': Random select N exmaples for plotting
