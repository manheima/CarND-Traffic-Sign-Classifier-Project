# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: report-images/TraingingSetClassifications.png "Visualization"
[image2]: report-images/ColorVsGrayscale.png "Grayscaling"

[image4]: traffic-signs-web-images/BumpyRoad_22.jpg "Bumpy Road"
[image5]: traffic-signs-web-images/SlipperyRoad_23.jpg "Slippery Road"
[image6]: traffic-signs-web-images/NoEntry_17.jpg "No Entry"
[image7]: traffic-signs-web-images/WildAnimalsCrossing_31.jpg "TWild Animals Crossing"
[image8]: traffic-signs-web-images/Yield_13.jpg "Yield"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


Here is a link to my [project code](https://github.com/manheima/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) but converted to grayscale this is (32,32,1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of samples of each classification in the training data set. 


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale because I tried both grayscale and color and found that the grayscale performed slightly better. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10X16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| output 400       								|
| Fully Connected	    | output 120                                    |
| RELU					|												|
| Fully Connected	    | output 84                                     |
| RELU					|												|
| Fully Connected	    | output 43                                     |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer which is an alternative to stochastic gradient descent. I used a batch size of 64 with 75 epochs, and a learning rate of 0.00093. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.951
* test set accuracy of 0.932

I chose to use the LeNet architecture. I believed it would be relevant to the traffic sign application because it worked well for detcting hand drawn numbers in the lab assignment. The final model's accuracy on the training, validation and test set provide evidence that the model is working well but is a little overfitted because the test set had a lower accuracy than the validation set. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

*Bumpy Road*

![alt text][image4] 

*Slippery road* 

![alt text][image5] 

*No Entry*

![alt text][image6] 

*Wild Animals Crossing*

![alt text][image7] 

*Yield*

![alt text][image8]

The second image may be difficult to classify because the sign is not centered, its is slightly angled, and it is a little far away.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road       		| Bumpy Road  									| 
| Slippery road 		| Traffic signals 								|
| No Entry				| Speed limit (100km/h)							|
| Wild Animals Crossing	| Wild Animals Crossing					 		|
| Yield     			| Yield  						            	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares less favorably than the accuracy on the test set of 93%. This is likely due to my model being a little overfit to the dataset and to me picking difficult images from the web to classify.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Bumpy Road    								| 
| .69     				| Slippery road									|
| 1.00					| No Entry										|
| 1.00	      			| Wild Animals Crossing					 		|
| 1.00  			    | Yield     					        		|


For the second image the algorithm predicted a 30% chance of classifying the road sign as a "Priority road". However, this was also not the correct classification of "Slippery road". I think the Nueral net was a little overfitted. 


