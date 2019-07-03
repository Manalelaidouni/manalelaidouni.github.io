---
layout: post
title: 'SSD : Understanding single shot object detection'
date : 2019-06-25
tags: [object detection, SSD]
permalink: Single shot object detection.html
categories: 
  - Jekyll
excerpt: A thorough explanation of the inner workings of SSD, its key contributions to performing faster than state of the art detectors (YOLO) while being as accurate as Faster R-CNN.
---
### Evolution of object detection algorithms leading to SSD

Classic object detectors were based on sliding window approach (DPM), this was computationally intensive due to the exhaustive search, but they were quickly rendered obsolete by the rise of region proposals with (R-CNN, Fast R-CNN), this approach introduces a selective search algorithm for finding potential objects, it performed relatively better, but it had its limitations, since the search algorithm is a decoupled process from training it would generate false positives and forward them to training, this combined with the complex pipeline made it computationally expensive as well as slow with 0.4 frames per second (fps) for Fast R-CNN.

Faster-RCNN solves the disjoint pipeline problem by replacing the external region proposal algorithm with an embedded Region Proposal Network (RPN) making it a unified network, this achieved 73.2 mAP on Pascal VOC2007 test set, but it remains significantly slow for real-time detection applications with 7 fps.

The next major improvement  in object detection history was to drop region proposals altogether and predict bounding box and confidence class scores directly from the image, one stage detectors, this structural simplicity allows for real-time predictions. Introducing You Only Look Once (YOLO) a simplified network processing 21 fps and achieving an accuracy of 66.4 mAP on Pascal VOC2007, you might observe that by increasing speed we risk of decreasing the detector's accuracy, this reframes the detection question to how further improve the speed vs accuracy trade-off.

This is where SSD is introduced performing faster namely to YOLO with 46 fps and a competitive accuracy to Faster-RCNN with 74.3 mAP.

In this post I explain the inner workings of SSD single shot detector, and its main contributions to performing faster and more accurately even with smaller sized input images.

### Network design

Single Shot Detector (SSD) uses a unified network of two parts, the base network leveraging a pre-trained VGG16 network on ImageNet, truncated before last classification layer to extract high level features, it then converts FC6 and FC7 to convolutional layers. 

Next SSD extends this base network by adding auxiliary structure of convolution feature layers of progressively decreasing resolution Conv8_2, Conv9_2, Conv10_2 and Conv11_2.

![SSD architecture](/assets/img/pexels/Image.png)

{:.image-caption}
*SSD architecture ([source](https://arxiv.org/pdf/1512.02325.pdf))*


### Forward pass workflow

The base network processes the input image to extract higher level features, it then passes through the multi-scale feature maps.
 
For each of the following feature maps  : conv4_3, conv7 , conv8_2, conv9_2, conv10_2, and conv11_2 :

SSD sets K defaults boxes (e.g. 6) of different scales and aspect ratios in a convolutional manner over each cell of the feature map. The default boxes are prior boxes chosen from the training set to match the ground truth bounding boxes.
 
> Using default boxes significantly facilitates the regression task since the network predicts bounding boxes from priors that are of similar shape to objects in the training set instead of learning them from scratch.

SSD then passes a 3x3 filter over the feature map, and for each default box, this filter simultaneously predicts 4 bounding box coordinates relative to the default box  ∆(cx, cy, w, h), i.e. the offsets to the default box to fit the ground truth box better, additionally for each default box the filter outputs confidence scores for all object categories (c1, c2, · · · , cp), this is processed simultaneously hence the name single shot.

![SSD architecture](/assets/img/pexels/SSD.jpg)


### Default boxes and multi-scale feature maps

The reason SSD predicts offsets and confidence scores over multiple feature maps of different resolutions is to be able to detect objects of different scales, more precisely, lower resolution feature maps detect larger objects in the image, this is partially due to the fact that semantic features of small-scale object are lost at these upper layers. As a case in point, the 8 x 8 feature map (b) detects the cat in image (a), the smaller object, while the lower resolution feature map of 4 x 4 (c) detects the larger object, the dog.

![Feature maps in SSD](/assets/img/pexels/FeatureSSD.png)

{:.image-caption}
([source](https://arxiv.org/pdf/1512.02325.pdf)) 



>Some methods process the input image multiple times, each time with different sizes to detect objects of different scales, however SSD achieves the same purpose by using feature maps of different resolutions while sharing parameters for all object scales and not be as computationally intensive.

Small deeper resolution feature maps detect high level semantic features where small-scale object features are lost, and since SSD uses progressively decreasing feature map resolutions,  it performs worse on small objects, however increasing the input image size particularly improves small object detection.

Furthermore, SSD sets K default boxes of different scale and aspect ratio over every location of different feature maps, in order to discretize systematically the search space for possible objects of different shapes in the image.

### Training

For the sake of comprehensiveness, I discuss the training process before covering the SSD objective function, because the latter is closely related to a matching strategy used in training, so please bear with me.

#### Matching strategy

During training, SSD aims at simplifying the learning process by selecting the best default boxes that overlap the most with each ground truth box instead of selecting the best one, then train the network accordingly, but what is the matching strategy that SSD uses?

For each ground truth box, SSD selects the best default box that overlaps the most with this specific ground truth box by finding the one that has the highest IoU (intersection over union), also referred to as Jaccard index.

1. For each ground truth box :

 - Compare ground truth box to default boxes & match it with the default box that obtains the highest IoU (intersection over union), also named Jaccard index.

SSD takes another step to simplify learning furthermore :

2 . For each default box :

 - Compare each default box to ground truth boxes, keep the ones that have an IoU > 0.5.

>Considering the large number of default boxes that SSD sets across multiple feature maps, after the above-mentioned matching strategy, the unmatched boxes would be of large number, so how does SSD deal with these unmatched boxes? 
>
>This brings us to the next section, introducing hard negative mining.

####  Hard negative mining : 
SSD mines the unmatched default boxes to be negatives examples , which are background images, and forwards them to training to make the classifier more robust, however this creates an overwhelming number of negative examples that produces a class imbalance, where the model learns the background images more than the more complex positive object categories.

Nonetheless, instead of using all negative examples, we can select just the best ones and forward them for training :

Sort the examples by their individual background confidence loss (i.e. individual cross entropy loss) in ascending order, then forward only the top ones for training, i.e. the negative examples of high confidence score, the ratio for each training batch is 3 negatives for 1 positive example.

####  Data augmentation

In addition to training on images with original input size, SSD samples patches that must overlap with ground truth boxes according to a Jaccard index of (0.1, 0.3, 0.5, 0.7, 0.7, 0.9) in order to systematically select patches that contain objects, additionally SSD samples few more patches randomly, these patches create a "zoom in" effect. All of the mentioned patches are then resized to a fixed size (e.g. 300 x 300 for SSD300) and flipped horizontally, in addition to some photogenic distortions (Random brightness, contrast, saturation, etc).

This overly extensive sampling strategy improves the model's performance significantly by 8.8  mAP on Pascal Voc2007 test set with a low resolution input of 300x300.

### Loss Function

Now addressing the training objective , SSD uses Smooth $$L1$$ loss to quantify the difference between the predicted box $$l$$ and ground truth box $$g$$ parameters $$(cx, cy, h, w)$$ where $$(cx,cw)$$ denote the center of the offsets to default box $$d$$ and $$h$$ and $$w$$ denotes its height and width respectively. The SSD paper along with Fast R-CNN and Faster R-CNN chose $$L1$$ loss to be their localization loss because it's more robust to outliers than $$L2$$ loss.

![Localization loss](/assets/img/pexels/loc_loss_SSD.png)

{:.image-caption}
*Localization loss ([source](https://arxiv.org/pdf/1512.02325.pdf))*



$$ \mathbb{x}{ij}^\text{p} $$ indicates the $$j$$-th matched ground truth box to the $$i$$-th default box for category $$p$$, $$\mathbb{x}{ij}^\text{p} $$ = 1 if there is a match according to the above-mentioned matching strategy and 0 otherwise, this makes the loss focus on the matched boxes and discard the others then train the model accordingly.

Additionally, the confidence loss is a cross entropy loss over $$c$$  categories :

![Confidence loss](/assets/img/pexels/Capture.PNG)

{:.image-caption}
*Confidence loss ([source](https://arxiv.org/pdf/1512.02325.pdf))*

The final loss objective is expressed as :

![Total loss](/assets/img/pexels/ssd_loss.png)

{:.image-caption}
*Final SSD loss ([source](https://arxiv.org/pdf/1512.02325.pdf))*

The localization loss could be greater than the loc loss, therefore the network would focus more on learning the localization task than the classification one, so we add a weighted term  $$\alpha$$	 to balance the losses so the model would focus on learning both of the tasks, and N is the number of matched default boxes.

### Inference time

Taking into account the large number of boxes SSD produces, it discards boxes with confidence score lower than 0.01, these are empty boxes, it then proceeds to perform non-max suppression (nms). When multiple boxes detect the same object, nms keeps only the best box per object.

Here is how it operates, for each class, nms selects the box with the highest confidence score and discards the remaining that overlap the most with this box, this is to keep only the best box per object, it uses a threshold of  IOU>=0.45.

The first filtering process combined with nms results in minimizing the number of boxes per image to 200.

### Performance analysis

* To further evaluate the effect of different default box shapes on performance, [Wei Liu et al](https://arxiv.org/pdf/1512.02325.pdf) removed default boxes with 1/3 and 3 aspect ratios, this yielded a decreased performance of 0.6 %, and by further removing the boxes with 1/2 and 2 aspect ratios, the performance decreased by 2.1 %.


* To investigate the effect of using feature layers of different resolution, Wei Liu et al proceeded by progressively removing each of the multi-scale layers and inspecting the effect of each one on detection accuracy, while also preserving the number of default boxes, this is accomplished by moving the boxes of the removed layer to the remaining layers and adjusting their scales to fit the tiling of default boxes. The following table shows the result of the experiment :

![SSD architecture](/assets/img/pexels/SSD-performance.png)

{:.image-caption}
*Effect of using different feature layers on SSD performance ([source](https://arxiv.org/pdf/1512.02325.pdf))*


> Analyzing the above table we see that using only conv7 layer decreased accuracy by a great margin of 11.9% mAP, emphasizing on the importance of using multiple feature maps of different resolutions.
 
* Interestingly enough, not using conv11_2 , the smallest resolution layer, increased performance by a small margin, this is perhaps due to small resolution feature maps being inadequate of small object detection.

* When using boxes of different scales and aspect ratios, the majority are bound to be located on the boundaries of  feature layers, they need to be handled carefully, therefore the authors chose to follow a technique used by Faster R-CNN by completely discarding these boundary boxes, and comparing the accuracy result based on that.

### Performance comparison

* The following is a scatter plot of speed and accuracy of the major object detection methods (R-CNN, Fast R-CNN, Faster R-CNN, YOLO and SSD300), needless to say, that for a fair comparison we use the same model setting (VGG16 as the base network, batch size of 1 and tested on Pascal VOC2007 test set). Note that YOLO and SSD300 are the only single shot detectors, while the others are two stage detectors based on region proposal approach.       

![SSD performance comparison](/assets/img/pexels/SSD-performance1.png)

{:.image-caption}
*SSD performance comparison ([source]( http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf))*


> SSD is the only object detector that achieves above 70% mAP accuracy while being real-time model with 46 fps.

* SSD  outperforms YOLO in terms of accuracy , while also being significantly faster with a margin of 25 fps.

* Faster R-CNN uses input images of 600x600, SSD achieves comparable accuracy to Faster R-CNN while using lower input size of 300x300. Now it's clear that using larger resolution inputs perform better in terms of accuracy, so let's see how SSD performs with larger inputs with SSD512


![SSD architecture](/assets/img/pexels/SSD-performance2.png)

{:.image-caption}
*SSD performance comparison ([source]( http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf))*

> Although SSD512 is not a real-time detector like Faster R-CNN, With larger inputs, it performs better than Faster R-CNN with an 80% mAP accuracy.

### Further notes

* To further increase real-time speed, consider trying a better base network, since SSD spends 80% of the forward pass on the base network.

* The authors proposed an extra data augmentation step to improve small-scale object detection, they referred to it by the "zoom out" operation as opposed to the "zoom in'' operation mentioned above by random cropping. The technique is to randomly place the standard sized image (e.g. 300x300) onto an up to 16 times larger canvas, then down scale it again to the standard size and forward them for training, according to the paper, this added technique improved accuracy by roughly 3% mAP.

>Since COCO contains smaller objects than its Pascal VOC counterpart, using the "zoom out" technique when training SSD512 on COCO dataset achieves an accuracy of 80% mAP.





### Conclusion 

Thanks to the major feature of combining default boxes and multi-scale feature maps in a unified framework, SSD performs faster than state of the art real-time detectors (YOLO) while being as accurate as Faster R-CNN, significantly improving the speed vs accuracy tradeoff.

However as we mentioned earlier, SSD has its limitations of detecting small objects due to the loss of their semantic features, this drawback is fairly remedied by a major feature introduced in the [RetinaNet paper](https://arxiv.org/abs/1904.02948) which is implementing a FPN (Feature Pyramid Network), that uses the deep high-level semantic feature maps to construct higher resolution feature layers thus detecting small-scale objects better. The paper introduces this concept along with an innovative approach to address the class imbalance problem, nonetheless I will explain the RetinaNet paper in a future post as a part of addressing single stage object detectors. 
