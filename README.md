# Training a face recognizer using OpenCV, TensorFlow, and Keras
### Introduction

Last year, I was inspired by a tutorial that taught how to create a face recognizer using OpenCV's built in LBPH recognizer. That project can be found [here](https://github.com/MagicTurtle2203/Face-Recognition-Test). In short, I grabbed pictures of 3 people, trained the face recognizer, and then tested the recognizer on different pictures of those 3 people. I wasn't very scientific about the process, as I just wanted to see if it would work with data that I provided on my own, and so I didn't record any statistics like how long it took to prepare the data and train and how accurate it was in the end. Overall, though, I was happy with myself that I was able to get it to work and that I had something to show for all my efforts. 

This new project was inspired by the first few videos in sentdex's tutorial series "Deep Learning basics with Python, TensorFlow and Keras", which can be found [here](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN). In it, he trains a convolutional neural network using Keras's Sequential model to distinguish between pictures of cats and dogs. That reminded me a lot of my previous project, and so I decided to try implementing this again with my own data and with my own twists on the project. I had tried to work with TensorFlow in the past, but back then I did not have as much experience with programming in general and so those attempts never amounted to much.

### Libraries used

Aside from Python's standard library, I had to install several libraries for this project. They were:
1. opencv-python
2. tensorflow-gpu (tensorflow)
3. matplotlib

### Collecting the data

For my previous project, I used various pictures of members from the K-POP group DREAMCATCHER which were all obtained from various fansites. Those images were all hand-collected and put together by me and that dataset was admittedly not very large. This time, I all of my pictures were downloaded from a Google Drive, link [here](https://drive.google.com/open?id=0B4f99lqreamSOVRwYU9NSWo3eU0), in which Discord user Alambi put together every picture of the group from fansites, promotional material, etc. To create the dataset used for this project, I downloaded every single image of the members Gahyeon and Handong (the same members from the previous project) in the years 2017-2018 from the `_pictures` folder and then spent a few hours going through every picture to make sure that there was only one face in each picture and that the members were clearly visible in their pictures. After checking again later, I found that I did manage to miss a few images that contained more than one member's face clearly visible, but I decided to leave those in. In the end, I ended up with around 10,000 images in total for both members. This amounted to about 13.7 GB worth of pictures. I also downloaded about 400 pictures from other sources to test the accuracy of each model after they had been trained, which amounted to about 430 MB worth of pictures. 

### The main project

The project was carried out in 3 parts, each documented in its own notebook. The notebooks are numbered so that they can be accessed easily, but they can also be accessed with the links down below.
1. Part 1 can be found [here](/1.%20dc_learning.ipynb)
2. Part 2 can be found [here](/2.%20dc_learning_haar_cascade.ipynb)
3. Part 3 can be found [here](/3.%20dc_learning_caffe.ipynb)
