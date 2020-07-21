# Road_Intersection_Classfify
Using CNN to classify road intersections, including datasets prepare and image operation
> Author: To_Fourier, CiWei
> Maybe as you guessed, To_Fourier is another name of QiuMingZS. 

# 1.File Description:
Network.py:           design of our cnn network
datasets_prepare.py:  help you to build your own datasets
image_pre_process.py: make images ready to be send to the network
train.py:             train the designed network
test.py:              get the result of road intersection classification

# 2.Network Description
conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> pool3 -> fc1 -> fr2 -> fc3 -> softmax

# 3.Functios:
Pool: max_pool
Activate: ReLU

# 4.Written at the end:
If you have any questions, welcome to raise an issue!
