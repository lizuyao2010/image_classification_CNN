### Thought:
* Pull all the data from endpoint
* Preprocessing the data 

 	Resize the image into a square image with dimension 224,224,3 

	Feed image into VGG Net to get VGG features(a 4096 dimension vector) 

* Feed pretrained VGG features and description feature(bag of words) into a softmax classifier as baseline

	VGG Net: http://www.cs.toronto.edu/~frossard/post/vgg16/

### Code:
	https://github.com/lizuyao2010/image_classification_CNN

### Data:
	
	data size: 33000

	description not null: 18163

	data split: 10% as test 

	Num of classes: 69 

	Vocab size: 3505 

### Examples


```
python clf.py
```

### Results:

labels shape(33000,69)

Features	|  Training Accuracy  |  Testing Accuracy  
------------|---------------------|-------------------
Text     	|  0.54	              |  0.52		          
Image     	|  0.94               |  0.72		          
Text+Image  |  0.98               |  0.76		       

main categories:

labels shape(33000,9)

Features	|  Training Accuracy  |  Testing Accuracy  
------------|---------------------|-------------------
Text     	|  0.56	              |  0.56		          
Image     	|  0.94               |  0.85		          
Text+Image  |  0.98               |  0.89		

### remove null descriptions:

labels shape(18163,65)

Features	|  Training Accuracy  |  Testing Accuracy  
------------|---------------------|-------------------
Text        | 0.97     			  |  0.92
Image       | 0.93     			  |  0.78
Text+Image  | 0.97                |  0.82		 

main categories:

labels shape(18163,9)

Features	|  Training Accuracy  |  Testing Accuracy  
------------|---------------------|-------------------
Text     	|  0.99	              |  0.98		          
Image     	|  0.95               |  0.85		          
Text+Image  |  0.99               |  0.93	
