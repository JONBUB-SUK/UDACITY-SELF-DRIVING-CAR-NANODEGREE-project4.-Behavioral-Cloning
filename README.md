# UDACITY-SELF-DRIVING-CAR-NANODEGREE-project5.-Behavioral-Cloning

# Introduction

The object of this project is drive car simulation autonomously by using deeplearning

Simulation provide two modes (training , autonomous) and 3 cameras are attached in center, left side, right side of car

Fistly, I had to done training mode at least 1 cycle of road

By driving test mode, simulator saved my steering action in csv file

So after one lap, I've got important information that what kind of background I turned steering

After one lap, there are almost 24,000 images (including left & right images)  and its steering angle informations

I used KERAS deep learning framework to my model learn

After trying various deeplearning models and methods, finally my car drove so well, even not get out of center line


# Background Learning

* Keras
 - How to handle Keras to use Deep Learning
 
* Transfer Learning
 - History and characteristics of various CNN architectures
 - About IMAGENET
 - AlexNet
 - VGG
 - GoogLeNet
 - ResNet
 


# Approach

1. Just 1 perceptron and Lambda Normalization (only using center camera images)

I did not expect good result

I thought car trained by this model will just stagger left and right

But unexpectively it somehow drives well

Of course it need many reinforcement

Its training data loss was 1.963 just after 1 epoch

(LOSS FUNCTION SCREENSHOT)


2. Using LeNet architecture (only using center camera images)

In fact, I expected this result a little, because its LeNet!!

I used LeNet architecture and parameters trained by IMAGE NET

And as a result, it drove somehow well

It could drive through curve but at the bridge, it drove to bridge wall and could not escape

Maybe its because it doesn't have data when get out of line

Its training loss was 0.0156 after 5 epochs

Loss is far less than first model!


(LOSS FUNCTION SCREENSHOT)


3. Adding left, right camera images to LeNet

This code have at least 2 benefits 

Firstly, We can get 3 times more data - Almost 8000 images 24000 images 

Secondly, By adding images leaning toward left, right we can train car returning to center

→  Use correction_factor plus minus steering angle

And surprisingly, by just adding left,right camera images made car drive through bridge well!

I thought why

Without left, right camera images, car cannot espcape when it get out of center or road or course

Then by adding left,right camera images, it trained how to escape by turning steering wheel

Assigning larger correction_factor may makes car turn more dynamic

As a result, training loss was 0.0097 after 3 epochs


(LOSS FUNCTION SCREENSHOT)


4. Adding cropping2D to LeNet

Just adding Cropping made car drive well!

I thought why

Cropped background and hood image is more simple than full image

So when car drives automatically and decides which image is equal to present condition, it is maybe more easy and precise

But this car cannot turn left at sharp curve

As a result, training loss was 0.016 after 3 epochs which is higher then before

But car drove better so I concluded getting smaller loss doesn’t make sure driving better


(LOSS FUNCTION SCREENSHOT)


5. Using NVIDIA CNN architecture

I just used architecture published in NVIDIA homepage they used when used their autonomous car

And not a surprisingly, it drove so well

It finished one lap without even escaping from center line

As a result, training loss was 0.014 after 3 epochs


(LOSS FUNCTION SCREENSHOT)


6. Adding a generator

Fidding all data at once is borne to GPU, so devide data by fit_generator

I devided by 32 data at one batch but it makes making result more slow

So I decided not to use ganerator

It gave me a challenge why it did slow down



# Results

(NEED NVIDIA ARCHITECTURE PICTURE)


# Conclusion & Discussion

1. data set

In fact, at starting this project, my intent was gathering data from 1 normal lap, 1 inverse lap, returning from out of line,

advanced track

But maybe because my laptop graphic card is not enough to play simulation, it works so slowly so that I cannot drive normaly

So I had to use only provided driving data

I wonder how it has different result if I used data I intended

Additionaly, even though I did not train data escaping from out of line, how can it drove so well?

Maybe that's because it did not go even out of center line

If it drove out of line, it cannot return to line, because it has no data to return


2. greatfulness of pre-trained architecture and parameters

I have got trouble trying various model to drive normally (I think it maybe easy)

And just using varified NVIDIA's architecture and parameter, it drove surprrisingly well

That is trained and tuned at real environment and it worked so well at similar situation


3. Normalization

At least in this project, maybe at deep learning problem in images, normalization is very important

It's feeding data consists of 0-255 and even it have 3 channels

So even after passing 1 level, multiplying weight, it has so large difference

It has effect to emphasize difference between neighbor pixels

In effect, doing normalization or not hadbig difference


4. About cropping

Overfitting is important subject in deep learning

Large data set is not always better in that regards

If I used not cropped image in this project,

model learned data from image that has background, and it think I had to turn left exactly in this situation (of course it cannot think)

So by apply cropping, it can work more generally


5. Small parameters

At least in self driving car, size of parameter is important

Because large size of parameter needs long time to calculate in real time

So I would like to revisit implementing this project to make less parameters


6. Discarded data

(jeremy-shannon)


7. About velocity

In this project, I cannot control velocity

The only factor I can control was steering angle at background image

But thinking in real driving situation, if someone want to sharp turn, he may firstly slow down car speed

So fundamentally this simulator has limit to sharp turn

If dealing with especially sharp turn by using this model,

it will need other method like enlarge steering angle parameters..










