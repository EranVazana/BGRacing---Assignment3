
# Assignment 3: Cone Detection Using AI

Hello and thanks for looking at my submission for the cone detection assignment!

Looking further to join the team to continue exploring the computer vision world.

Example result:

![Alt Text](https://s5.ezgif.com/tmp/ezgif-5-0383d8a8fd.gif)
<img src="[url](https://s5.ezgif.com/tmp/ezgif-5-0383d8a8fd.gif)" alt="Alt text">

## Acknowledgements

I trained the project YOLOV8 model for detecting cones using this dataset:

 - [Cones Dataset](https://universe.roboflow.com/yuval-k/cones-detection-k0i6h/dataset/2)


## Installation

The Cone Detection use both the OpenCV and Ultralytics libaries: 
```bash
  pip install ultralytics 
  pip install opencv-python
```
The repository containes the pre-trained Cone Detection model,
if you choose to train it on another dataset, make sure you have the pytorch libary:

```bash
  pip install torch
```

To install pytorch with the nvidia CUDA toolkit i recomment to look at:

https://pytorch.org/get-started/locally/
    
## Usage

Simply run the following in the repository terminal: 

```python
python -m ConeDetection.py
```

