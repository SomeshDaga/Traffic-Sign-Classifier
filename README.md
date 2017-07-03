# **Traffic Sign Recognition** 

This is a writeup for the Traffic Sign Recognition project as part of the Self-Driving Car Nanodegree program provided by Udacity.

[//]: # (Image References)

[sign_histogram]: ./hist.png "Dataset Visualization"
[sign-caution]: ./german-traffic-test/resized_caution.jpg "Caution Traffic Sign"
[sign-no-entry]: ./german-traffic-test/resized_no_entry.jpg "No Entry Traffic Sign"
[sign-ahead-only]: ./german-traffic-test/resized_ahead_only.jpg "Ahead Only Traffic Sign"
[sign-stop]: ./german-traffic-test/resized_stop.jpg "Stop Traffic Sign"
[sign-yield]: ./german-traffic-test/resized_yield.jpg "Yield Traffic Sign"

---

## Data Set Summary & Exploration

The basic parameters of the dataset/images was obtained using in-built Python functions. The parameters are listed below:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The no. of traffic signs per class was plotted to show the distribution of the dataset:

![][sign_histogram]

## Designing and Testing the Model Architecture

### 1. Techniques used for preprocessing the Traffic Sign Dataset

The first step of the preprocessing stage was to normalize the pixel values (`0-255, type: uint8`) to float values ranging between -1.0 and 1.0. This is primarily because of the Neural Network applying activation functions which are only meaningful for small value ranges. 

The next and final step of the preprocessing was to convert the images to grayscale. There were 2 reasons for doing this:

- Less weights and training time for Neural Network since images shrink from 3 channels (RGB) to just 1 channel.
- Based on training under RGB vs Grayscale, Grayscale seemed to give comparable/slight better results. I believe this may be because of higher uncertainties in classification when using RGB channels simultaneously (high variation in any of these channels might result in misclassification) in the convolution layers.

### 2. Choosing the Convolutional Neural Network Architecture

The final architecture mostly followed the design of the LeNet 5 Architecture explored in the Convolutional Neural Networks segment of the course. This was deemed acceptable because a validation accuracy of ~0.89 was achieved without implementing any of the robustness/improvements.

In order to bump the validation accuracy above 0.93, the following measures were incorporated into the Neural Network architecture:
- Dropouts between layers with a keep probability of 0.8 across all dropouts
- Using `tanh` instead of `relu` activations since it provides a more varied output, especially for normalized pixel values less than 0.

My final model consisted of the following layers:

| Layer         		| Operation |   Description	        					| 
|:---------------------:|:--------------:|:-------------------------------:| 
| Layer 1 | Input         		| 32x32x1 Grayscale image   							| 
|| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
|| Activation					|	`tanh` function											|
|| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
|| Dropout	    | `keep_prob = 0.8`      									|
| Layer 2 | Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
|| Activation					|	`tanh` function											|
|| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|| Dropout	    | `keep_prob = 0.8`      									|
|| Flatten     | Flattened the array, output 1x400 |
| Layer 3 | Feed forward | Multiplied by weights and biases. outputs: 1x120|
|| Activation | `tanh` function |
|| Dropout | `keep_prob = 0.8` |
| Layer 4 | Feed forward | Multiplied by weights and biases. outputs: 1x84|
|| Activation | `tanh` function |
|| Dropout | `keep_prob = 0.8` |
| Layer 5 | Feed forward | Multiplied by weights and biases. output `Logits`: 1x43|

 
### 3. Discussion of Hyperparameters used to train the model.

The following parameters were used to train the model:
- `EPOCHS: 10`
- `BATCH_SIZE: 128`
- `LEARNING_RATE: 0.001`
- `TRAIN_KEEP_PROB: 0.80`

An epochs of 10 was used primarily due to computation time required to train the neural network. Since, the neural network was trained on a local CPU and not a GPU, the aim was to limit the number of epochs to be able to run within a duration of 5 minutes. Moreover, it was seen that the validation accuracy w.r.t epoch number would cease its monotonic behaviour at around the 8th epoch, signalling that it was sufficiently trained.

The learning rate was initially varied in the range of 0.0001 to 0.01, but was found to either very gradually change the validation accuracy with epochs or worsen the validation accuracy (due to weights changing by too big a factor). 0.001 seemed to be the sweet spot for the values tested.

The `keep_prob` value for the dropouts was also varied in the range of 0.7-0.95. It was observed that a lower `keep_prob` would typically result in a lower validation accuracy for the first few epochs, but the accuracy would rise more steeply with increasing number of epochs. Given that we were working with a low number of epochs (i.e. 10), the aim was to not have the value to be too low (rate of increase does not compensate for low initial accuracy) or too high (higher initial accuracy but very low rate of increase).

The optimizer used for reducing the softmax cross entropy, was the `Adam Optimizer` based on its suitability for the given type of problem (as reference to in the abstract [here](https://arxiv.org/abs/1412.6980)).

### 4. Results and Rationale behind Final Model Architecture

The use of Convolutional Neural Network architecture was a given because the aim was to identify/classify a traffic sign in any part of the images in the dataset. The final architecture chosen was very similar to the LeNet 5 model used in the Convolution Neural Networks part of the Udacity course. It was primarily chosen because we had proven its effectiveness as a classifier on image datasets.

Given the aim of establishing a minimum accuracy of 0.93 on the validation dataset, it was reasonable to expect measures such as implementing dropouts and using more distinct mapping activation functions would be able to bump up an accuracy of 0.89 by a few percentage points. The parameters that were tuned were mostly the `learning_rate` and the `keep_prob` to achieve the desired results. The tuning of these parameters was based on trial and error, seeing how the validation accuracy changes with epochs for a given set of values and making sense of why it might do that. I also initially tried to vary the `standard deviation` of the randomized weights generated, but this yielded catastropic results for values more than 0.3 (accuracies in the range of 0.05-0.2).

My final model results were:
* validation set accuracy of 0.942
* test set accuracy of 0.927

## Test a Model on New Images

### 1. The five images chosen for the model test

Here are five German traffic signs that I found on the web:

![Caution][sign-caution] ![No Entry][sign-no-entry] ![Speed Limit 30][sign-ahead-only] 
![Stop][sign-stop] ![Yield][sign-yield]

Note that these images have been scaled down to 32x32 in order to be used with the model architecture described previously. This is why some of the aspect ratios seem off, as the majority of these original images were not of equal heights and widths. The images were scaled using the PIL Python API.

The potential reasons why a sign may not be classified correctly are:
- Warped image due to rescaling while not maintaining aspect ratio
- Not enough training data for the image chosen. Refer to the visualization of the dataset exploration to see which signs saw the most training data and which didn't
- Area covered by the signs in the image. The convolutions specified must be able to capture the sign adequately. If a sign is too small, the convolution might capture other parts of the environment and not be able to capture the distinct features of the particular sign.

### 2. Model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Caution Sign      		| Caution sign   									| 
| Yield Sign     			| Yield Sign 										|
| Stop Sign					| Yield Sign										|
| No Entry Sign	      		| No Entry Sign				 				|
| Ahead Only Sign			| Ahead Only Sign      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.2%. Given that only 5 images were used, it is difficult to say with certainty that we will achieve the 94.2% target with new images that may vary from those of the dataset.

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image (General Caution Sign), the model is very sure (99.7%) that the sign is a General Caution sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Caution Sign   									| 
| .020     				| Traffic Signals 						  		|
| .000					| Pedestrians											|
| .000	      		|	 Road Work					 				|
| .000				    | Right-of-way at the Next Intersection      							|


For the second image (Yield Sign), the model is very sure (~99.9%) that the sign is a Yield sign. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         			| Yield sign   									| 
| .000     				| Ahead Only 										|
| .000					| Road Work											|
| .000	      			| Keep Left					 				|
| .000				    | No Vehicles      							|


For the third image (Stop Sign), the model guesses wrong and is not very sure about any of its top 5 choices. Unfortunately, the correct sign type is not even in its top 5 predictions.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .477         			| Yield Sign   									| 
| .324     				| Speed Limit 60 										|
| .076					| No passing for vehicles over 3.5 metric tons											|
| .036	      			| Ahead Only					 				|
| .027				    | Priority Road      							|


For the fourth image (No Entry Sign), the model is very sure (~99.6%) that the sign is a No Entry sign. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| No Entry   									| 
| .003     				| Stop 										|
| .000					| End of all speed and passing limits											|
| .000	      			| Turn left ahead					 				|
| .000				    | Turn right ahead      							|


For the fifth image (Ahead Only Sign), the model is very sure (~98.2%) that the sign is a Ahead Only sign. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .982         			| Ahead Only   									| 
| .006     				| Keep Left 										|
| .005					| Dangerous curve to the left										|
| .003	      			| Go straight or right					 				|
| .002				    | Road Work      							|




