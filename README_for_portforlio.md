# SELF DRIVING CAR NANODEGREE
# Project5. Behavioral Cloning

[//]: # (Image References)

[image1-1]: ./images/1.1_Percaptron,Lambda.JPG "RESULT1"
[image1-2]: ./images/2.LeNet.JPG "RESULT2"
[image1-3]: ./images/3.Center,Left,Right_images.JPG "RESULT3"
[image1-4]: ./images/4.Cropping.JPG "RESULT4"
[image1-5]: ./images/5.NVIDIA_architecture.JPG "RESULT5"

[image2-1]: ./images/NVIDIA_CNN_architecture.png "NVIDIA"

[image3-1]: ./images/Behavioral_cloning_gif.gif "RESULT GIF"

## 1. Introduction

The object of this project is to drive car simulator autonomously by using Deep Learning

Simulation provide two modes (training , autonomous) and 3 cameras are attached in center, left side, right side of car

Firstly, I had to done training mode at least 1 cycle of road

By driving test mode, simulator saved my steering action in csv file

So after one lap, I've got important information that at which kind of background I turned steering

After one lap, there are almost 24,000 images (including left & right images) with its steering angle informations

I used KERAS deep learning framework to make my model learn

After trying various Deep Learning models and methods, finally my car drove so well, even not getting out of center line


## 2. Related Study

#### 1) Keras

① How to handle Keras to use Deep Learning
 
#### 2) Transfer Learning

① History and characteristics of various CNN architectures
 
② About IMAGENET

③ AlexNet

④ VGG

⑤ GoogLeNet

⑥ ResNet
 

## 3. Details

#### 1) Just 1 perceptron and Lambda Normalization (only using center camera images)

I did not expect good result

I thought car trained by this model will just stagger left and right

But unexpectively it somehow drove well

Of course it need further reinforcement

```python
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
```

Epochs : 1

Training loss : 1.6953

Validation loss : 0.5141

#### 2) Using LeNet architecture (only using center camera images)

In fact, I expected this result a little, because its LeNet!!

I used LeNet architecture and parameters trained by IMAGE NET

And as a result, it drove somehow well

It could drive through curve but at the bridge, it drove to bridge wall and could not escape

Maybe its because it doesn't have data when get out of line

Its training loss was 0.0156 after 5 epochs

Loss is far less than first model!

```python
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Conv2D(10,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
```

Epochs : 5

Training loss : 0.0156

Validation loss : 0.0171


#### 3) Adding left, right camera images to LeNet

I added not only center camera images, but also right, left camera images

This code have at least 2 benefits 

Firstly, I can get 3 times more data - Before almost 8000 images to 24000 images 

Large amount of data have benefits not to be overfitted

Secondly, By adding images leaning toward left, right I can train car returning to center

`steering angle = correction_factor ± steering angle`

And surprisingly, by just adding left,right camera images made car drive through bridge well!

I thought why

Without left, right camera images, car cannot espcape when it get out of center or road or course

Then by adding left,right camera images, it trained how to escape by turning steering wheel

Assigning larger correction_factor may makes car turn more dynamic

As a result, training loss was 0.0097 after 3 epochs

```python
for line in lines:
    for i in range(3):
        correction_factor = 0.2
        source_path = line[i]
        file_name = source_path.split('/')[-1]
        current_path = '/opt/carnd_p3/data/IMG/' + file_name
        image = ndimage.imread(current_path)

        images.append(image)
        
        if i==0:
            measurement = float(line[3])
            measurements.append(measurement)
        elif i==1:
            measurement = float(line[3]) + correction_factor
            measurements.append(measurement)
        elif i==2:
            measurement = float(line[3]) - correction_factor
            measurements.append(measurement)
            
X_train = np.array(images)
Y_train = np.array(measurements)        
```

Epochs : 3

Training loss : 0.0097

Validation loss : 0.0114

#### 4) Cropping not useful part of images

Just adding Cropping made car drive well!

I thought why

Cropped background and hood image is more simple than full image

So when car drives automatically and decides which image is equal to present condition, it is maybe more easy and precise

But this car cannot turn left at sharp curve

As a result, training loss was 0.016 after 3 epochs which is higher then before

But car drove better so I concluded getting smaller loss doesn’t make sure driving better

```python
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(10,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15,(5,5), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
```

Epochs : 3

Training loss : 0.0161

Validation loss : 0.0202


#### 5) Using NVIDIA CNN architecture

I just used architecture published in NVIDIA homepage they used when making their autonomous car

<img src="./images/NVIDIA_CNN_architecture.png" width="300">

And not a surprisingly, it drove so well

It finished one lap without even escaping from center line

As a result, training loss was 0.014 after 3 epochs

```python
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(36,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(48,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
```

Epochs : 3

Training loss : 0.0144

Validation loss : 0.0222


#### 6) Adding a generator

Fidding all data at once is burden to GPU, so devide data by fit_generator

I devided by 32 data at one batch

```python
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    file_name = source_path.split('/')[-1]
                    current_path = '/opt/carnd_p3/data/IMG/' + file_name
                    image = ndimage.imread(current_path)
                    images.append(image)
                    
                    correction_factor = 0.2   # For using left&right camera image
                    if i==0: # Center camera
                        angle = float(batch_sample[3])
                        angles.append(angle)
                    elif i==1: # Left camera
                        angle = float(batch_sample[3]) + correction_factor
                        angles.append(angle)                        
                    elif i==2: # Right camera
                        angle = float(batch_sample[3]) - correction_factor
                        angles.append(angle)      

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D

# <This is third model using NVIDIA CNN architecture>
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(36,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(48,(5,5), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
                    steps_per_epoch = len(train_samples), 
                    validation_data = validation_generator, 
                    validation_steps = len(validation_samples), nb_epoch=3)

# model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model_generator.h5')
```

#### 7) Finally make simulator to run with trained data

```c++
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)
```


## 4.Results

This is a gif image for result video

![alt text][image3-1]


## 5.Discussion

#### 1) data set

In fact, at starting this project, my intent was gathering data from 1 normal lap, 1 inverse lap, returning from out of line,
advanced track

But maybe because my laptop graphic card is not enough to play simulation, it works so slowly so that I cannot drive normaly

So I had to use only provided driving data

I wonder how it has different result if I used data I intended

Additionaly, even though I did not train data escaping from out of line, how can it drove so well?

Maybe that's because it did not go even out of center line

If it drove out of line, it cannot return to line, because it has no data to return


#### 2) Greatfulness of pre-trained architecture and parameters

I have got trouble trying various model to drive normally (I think it maybe easy)

And just using varified NVIDIA's architecture and parameter, it drove surprisingly well

That is trained and tuned at real environment and it worked so well at similar situation


#### 3) Normalization

At least in this project, maybe at deep learning problem in images, normalization is very important

It's feeding data consists of 0-255 and even it have 3 channels

So even after passing 1 level, multiplying weight, it has so large difference

It has effect to emphasize difference between neighbor pixels

In effect, doing normalization or not hadbig difference


#### 4) About cropping

Overfitting is important subject in deep learning

Large data set is not always better in that regards

If I used not cropped image in this project,

model learned data from image that has background, and it think I had to turn left exactly in this situation (of course it cannot think)

So by apply cropping, it can work more generally


#### 5) Small parameters

At least in self driving car, size of parameter is important

Because large size of parameter needs long time to calculate in real time

So I would like to revisit implementing this project to make less parameters


#### 6) About velocity

In this project, I cannot control velocity

The only factor I can control was steering angle at background image

But thinking in real driving situation, if someone want to sharp turn, he may firstly slow down car speed

So fundamentally this simulator has limit to sharp turn

If dealing with especially sharp turn by using this model,

it will need other method like enlarge steering angle parameters..


#### 7) Convenience of Keras compared to Tensorflow

In fact at the before project using tensorflow, it was so strange concept like tensor and its usage

But unlikely tensorflow, Keras is so easy to use and intuitive to understand

After this moment I feel like using Keras at every deep learning project








