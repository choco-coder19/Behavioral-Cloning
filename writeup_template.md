#**Behavioral Cloning** 

##Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images/center.jpg "Normal driving"
[image3]: ./images/recover1.jpg "Recovery Image"
[image4]: ./images/recover2.jpg "Recovery Image"
[image5]: ./images/recover3.jpg "Recovery Image"
[image6]: ./images/flip1.jpg "Normal Image"
[image7]: ./images/flip2.jpg "Flipped Image"
[image8]: ./images/model.PNG "Model"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network similar to one described in [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from Nvidia.

As is described in line 53 to 79 from `process.py`, this architecture is composed by 5 convolutional layers, followed by 5 fully connected layers with dropout. ELU is used as activation, to introduce nonlinearities to the model. Adam optimizer is used, so no tunning is needed in learning rate. Keras lambda layer keeps the code simple, letting us to normalize the images directly. A cropping2d layer at the beggining selects the ROI.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after every fully connected layer.  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track on both circuits. The fact that the second one, not used in trainig, could be used by 1/5 means that our model is not overfitting. Also, image tranformations like flipping and driving backwards on the first circuit helps.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 80). Dropout values were set proportional with the layer size. The number of epochs was set to 20, but stops improving after a while.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and side cameras whith an angle correction to teach the car how to recover from a side.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model, due to the ability of this kind of systems to get information about images. After trying with some custom network architectures, I saw David's video and reference to the Nvidia paper, and it was great!

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My first model went always to the left, but flipping the images and changing stering sign make it more stable. 

Then I deal with the problem of the car crashing in the bridge and one curve. Getting more data and cropping it helped me to get my first complete lap!

My computer is not very powerfull, and my internet connection is very slow. The model was trained in an AWS instance (g2.2large for GPU usage with Tensorflow CUDA backend), but running the simulator in my computer with low graphics and size was a challenge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, but it could be improved for sure. 

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

* Input layer (160x320x3)
* Normalization layer
* Cropping layer (160x320x3 to 65x320x3)
* Maxpooling (65x320x3)
* Conv Layer #1 (5x5 filter, 8x40x5, 1805 param)
* Activation (ELU)
* Conv Layer #2 (5x5 filter, 4x20x5, 4505 param)
* Activation (ELU)
* Conv Layer #3 (5x5 filter, 2x10x5, 6005 param)
* Activation (ELU)
* Conv Layer #4 (3x3 filter, 1x5x3,  2883 param)
* Activation (ELU)
* Conv Layer #5 (3x3 filter, 1x3x3,  1731 param)
* Activation (ELU)
* Flatten layer (9)
* Dense layer (1164)
* Activation (ELU)
* Dropout layer
* Dense layer (100)
* Activation (ELU)
* Dropout layer
* Dense layer (50)
* Activation (ELU)
* Dense layer (10)
* Activation (ELU)
* Dropout layer
* Dense layer (Output, 1)

With a total of 150640 parameters, all of them trainable.


Here is a visualization of the architecture: 

![alt text][image8]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to do so. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I drove track one on the other direction for one lap.To augment the data sat, I also flipped images and angles thinking that this would help me fix my always turning left problem. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 15836 images. After flipping, I have twice this size.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10, with more it starts to overfit. I used an adam optimizer so that manually training the learning rate wasn't necessary.
