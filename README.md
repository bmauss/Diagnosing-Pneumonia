# Diagnosing-Pneumonia
Building a model that utilizes Deep Learning and Image Recognition to determine if patients have Pneumonia. 

## Objective
We've been tasked with creating a neural network model which utilizes image recognition to determine whether or not a patient has pneumonia based on their chest x-rays.

## Obtaining Data
Let's start with what we know.

Pneumonia is an infection in the lungs, specifically the alveoli, and can be caused by viruses, bacteria, or fungi.  Aside from blood tests and analyzing sputum, chest x-rays are also helpful.  This is because x-rays have difficulty penetrating through the fluid that builds up in the alveoli, creating a cloudy image.

The obstacle here will be that x-rays range in clarity due to noise or movement. Pneumonia also ranges in severity.  Some cases may be so acute that, to the untrained eye, they would appear normal.  In light of this and the health concerns around pneumonia we'll be aiming for a **model with high recall** and as **high of an accuracy we can achieve**.  We will be using recall as our metric since the worst case scenario with a **False Positive** is that someone is unnecessarily prescribed antibiotics and possibly steroids for a week.  The worst case scenario for a **False Negative** is **death**, if left untreated.

### About the data
The data was downloaded from Kaggle.com and, upon inspection, the files are not only divided into separate folders for positive and negative cases, but also split into training, test, and validation sets. This will save us some time during preprocessing.

How this affects preprocessing:

* We will still need to check for class imbalances

* As stated, we will be focusing on recall and accuracy for our metrics

* As this dataset only contains images, we won't need to worry about NaN's, placeholders, or feature engineering

The shape of each of our training sets are as follows: 

Training Set: (5216, 2)

Validation Set: (16, 2)

Testing Set: (624, 2)

This gives us a grand total of **5,856 images**. Under different circumstances, I would say that this split ratio is going to lead to model over-fitting (it still might if we're not careful). Though, given the nature of neural networks and the amount of noise that is in some of these images, we'll keep this **90/10 split** of the data.

### Class Imbalance
![Imgur](https://i.imgur.com/eqHoubj.png)

There is definitely class imbalance, but, due to the amount of noise in both the "Normal" and "Pneumonia" x-rays, this isn't necessarily a bad thing.  It could help our neural network distinguish between the two classes more accurately since it will have so many pneumonia cases to sift through.  We'll leave this as it is and move on.

## Baseline Model

We'll be utilizing a Convolutional Neural Networks (CNN) for our models. CNNs utilize "filters" to detect edges and other objects to summarize and shrink the data (if you choose). We'll also be able to add pooling layers which will summarize what the previous layer accomplished so that the model can build off of it in the next layer. Finally, we'll use fully connected (Dense) layers so that the neural network can do a thorough evaluation of the transformed (simplified) data.

```
np.random.seed(42)

model = models.Sequential()

# Due to the amount of noise in our data, we'll use a smaller filter of 3x3
# This time around we won't be using padding.
model.add(layers.Conv2D(50, (3,3), activation='relu', 
                        input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Conv2D(50, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(75, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(75, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='sgd',
             metrics=['acc'])
```
The model's architecture is as follows:
* 4 Hidden Layers (HL)
* Model accepted normalized images with a resolution of 100 x 100 pixels
* Each Convolutional layer utilized:
* Convolution filter sizes of 3x3
* MaxPooling2D with 2x2 filters
* All Hidden Layers utilized ReLU activation
* HL 1 & 2 contained 50 filters each
* HL 3 contained 75 filters
* HL 4 : Dense layer with 75 neurons

It's simple, but it's meant to be a plain, vanilla baseline to build off of.

After running this model through 75 epochs, we ended with a **Test accuracy** of **72%**.  Here's a look at our metrics throughout the training process:

![Imgur](https://i.imgur.com/vxDDKbk.png)

The validation metrics are fluctuating a little too much for my liking, especially the validation loss. This could be that there weren't enough epochs, but it could also be that there just isn't enough validation data for the model to find a pattern with.

The plateaus of the validation accuracy and the spikes in loss dropping more as time passes seemed to indicate that the model might converge if given additional time.  With that in mind we ran the model through 150 epochs to see what happened.

![Imgur](https://i.imgur.com/X2HMq2t.png)

A ton of variance and an overfit model was the result. It appears that with more epochs, the results in the training and validation sets look like they are converging, at first. As they iterate through more epochs, however, they begin to diverge.  

## Gradient Tuning

After a lot of trial and error with dropping nodes and filters, the addition of padding, changing input size, BatchNormalization, Dropout layers, regularization, and data augmentation, we found the following architure to work best.

### The Iteration Process

We chose to utilize a **kernel initializer** to give our model some weights to start with.  **He_normal** chooses weights from a truncated normalized sample and then the optimizer would tune those weights during backward propogation.  This He_normal also had a reputation for working really well with our **ReLU** activation function.

We changed our optimizer from Stochatic Gradient Descent (Gradient Descent with Momentum) to Adam, since Adam is highly optimized and has proven to be more versitile than SGD.

Due to how eager the model was to overfit, we chose **L2 Regularization** to place heavier penalties on larger weights.  

This proved to not be enough, however, on its own.  In fact, during our testing and retesting of various models, we found that data augmentation and regularization had **very little effect if you went with one or the other**.  It was **only when using both** that we started to see an effect.  You'll find our data augmentation parameters below:

```
aug_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=30,
                                width_shift_range=0.15,
                                height_shift_range=0.15,
                                shear_range=0.1,
                                zoom_range=0.1,
                                vertical_flip=True,
                                fill_mode='nearest')
```

These parameters would increase the data by rotating, shifting, cutting off, zooming in on, and flipping the images at random.

The initial result of these changes was that the model was a model that was underfit.  From there, we simply had to increase the complexity of the model.  Then, to reduce the risk of overfitting again, we cut the number of epochs down to 40.  So we added more filters and nodes to the respective layers and ended up with the following model: 

```
np.random.seed(42)

model_augreg4 = models.Sequential()

model_augreg4.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', 
                        kernel_initializer='he_normal',
                        kernel_regularizer='l2',
                        input_shape=(128, 128, 3)))
# model.add(layers.BatchNormalization())
model_augreg4.add(layers.MaxPooling2D((2,2)))


model_augreg4.add(layers.Conv2D(128, (3,3), padding='same', 
                        kernel_initializer='he_normal',
                        kernel_regularizer='l2',
                        activation='relu'))
# model.add(layers.BatchNormalization())
model_augreg4.add(layers.MaxPooling2D((2,2)))

model_augreg4.add(layers.Conv2D(256, (3,3), padding='same', 
                        kernel_initializer='he_normal',
                        kernel_regularizer='l2',
                        activation='relu'))
# model.add(layers.BatchNormalization())
model_augreg4.add(layers.MaxPooling2D((2,2)))

model_augreg4.add(layers.Flatten())
model_augreg4.add(layers.Dense(64, kernel_initializer='he_normal',
                        kernel_regularizer='l2',
                       activation='relu'))
# model.add(layers.Dropout(0.2))
model_augreg4.add(layers.Dense(1, activation='sigmoid'))

model_augreg4.compile(loss='binary_crossentropy',
             optimizer='Adam',
             metrics=['acc'])
```

(Note: We had the BatchNormalization and Dropout layers commented out from previous models and intended to use them if we saw overfitting again)

## The Results

![Imgur](https://i.imgur.com/0qan4dG.png)

Test accuracy soared up to 85% and our loss was at it's lowest! 

We attempted to tune the hyperparameters, but this only resulted in the model vacilating between underfit and overfit.

## Transfer Learning

Before moving on to visualizing intermediate activations, we tried out transfer learning.  We chose the Xception model, not only for it's high performance, but because it included Depthwise Convolution layers.  Depthwise Convolution breaks the filter process into two separate stages.  The result of this is that the number of multipliers/parameters drops considerably, reducing model complexity and increasing efficiency. 

The results were, unfortunately, lack luster.

![Imgur](https://i.imgur.com/FrFuUr0.png)

Perhaps 20 epochs wasn't enough.  Or maybe the ImageNet that Xception was trained on didn't have a lot of x-ray images.  Either way, the test accuracy dropped to 75%.  

## Visualizing Intermediate Activations
We put two images through the model and visualized the outputs after each activation (or at least as many as our hardware could handle).  For the sake of brevity, we'll show the results for our pneumonia x-ray sample.

Original Sample Post-Preprocessing:

![Imgur](https://i.imgur.com/8VS0AcO.png)

### First Layer

![Imgur](https://i.imgur.com/eihRECI.png)

You’ll notice that the primary focus of the first layer in both cases is detecting the edges. You can see it finding the silhouettes of the body and ribcage in each image, setting them apart from the background.

### The Second Layer

![Imgur](https://i.imgur.com/T2zJLLK.png)

The images have become slightly more fuzzy as a result of the filters in the convolution layers.  We’re seeing less filtering out the foreground noise. Instead, the model appears to be searching more in the middle. Images that used to just be a shadow against a light screen have spots where the background is peeking through.

We’re also seeing more instances where the contrast is either turned up (creatings more crisp images) or turned down (creating fuzzy images).

### The Third Layer

![Imgur](https://i.imgur.com/7moOTyV.png)

The first is a “standard” image, with the surface noise.  In the second image, the colors are inverted, and the contrast is turned up to bring out the background. If you look at the pneumonia sample, this actually REALLY highlights the noise created by the virus.  The final picture, the contrast is returned to normal. This, again, highlights the pneumonia, while also helping the model distinguish it from the rest of the body by showing off the translucent nature of the virus.

### Findings

We can see that in addition to identifying general "cloudiness", the model inspects the separate levels of intensity so that it can, in essence, "peel away" layers of the image. This allows it to filter out noise from skin silhouettes to actual bones.

Why would you want to filter out this the rib cage if you're looking for cloudiness in the lungs? Well, the noise created by the pneumonia will "bleed" into the background. Essentially, the more concentrated noise present, the greater the general pixel intensity in all "layers" of the image, even in the background, causing what would look like deformations.

The model also cross-references this information by inverting the pixel intensities (essentially creating a "negative" of the image). This really highlights the cloudiness in images, even in lower resolutions and serves as a cross-validation of sorts.
