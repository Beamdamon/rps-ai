import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os

#API LIbrary Setups - Damon Beam - Expected Time 6 hours - Time Taken 9 hours
#I decided to try something pretty foreign to me this time. I think overall, it was a great learning experience. If I was going to do it again, I feel like I would have a much
#better understanding of how to do so. The time taken comes from figuring out how to actually get everything working, as well as learning what each function does, and how changing
#the variables can change the AI. Overall, I think it turned out decent enough, as it was able to predict rock, paper, or scissors decently well. I went over the expected time likely
#due to the trouble I had for the AI to recognize scissors, and getting that figured out.

#Code used from both:
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=it1c0jCiNCIM
#As well as ChatGPT to generate and help understand certain functions

#Training and test datasets from Tensorflow 
#Learning Dataset https://storage.googleapis.com/learning-datasets/rps.zip
#Testing Dataset https://storage.googleapis.com/learning-datasets/rps-test-set.zip 

trainDataDir = "rps"
testDataDir = "rps-test-set"

#Preprocessing of images using cv2 - sets image size to 150x150 and num of symbols (0 = paper, 1 = rock, 2 = scissors). 
#Sets at 32-bit and scaled for storage. Used for non dataset images.
imageSize = (150, 150)
numClasses = 3 

def preprocessImage(image):
    image = cv2.resize(image, imageSize)
    image = image.astype("float32") / 255.0 
    return image

#Loads the dataset from rps and rps-test-set and puts them in their respective array. Example: rps -> 0 -> paper.img -> images array
def loadDataset(dataDir):
    images = []
    labels = []
    
    for symbolLabel in range(numClasses):
        symbolDir = dataDir + "/" + str(symbolLabel)
        
        for imageName in os.listdir(symbolDir): #returns list of directory
            image = cv2.imread(os.path.join(symbolDir, imageName)) #
            image = preprocessImage(image)
            images.append(image)
            labels.append(symbolLabel)
    
    return np.array(images), np.array(labels)

# Loads and preprocesses the training and testing datasets
trainImages, trainLabels = loadDataset(trainDataDir)
testImages, testLabels = loadDataset(testDataDir)

#Model construction - code from Tensorflow to create a machine learning neural network.
#Conv2D - detects features using convolutions. Number is the number of fuunctions.  64/128 filters of 3x3.
#Pooling - groups up pixels in images to group of pixels in an image and filter the smaller ones out. Reduces size of image while maintaining image features. 2x2 set of pixels
#Flatten - flattens the input to go into the DNN
#relu - rectified linear unit - returns a value if greater than 0, filters. 
#softmax- Picks the biggest number in a set, probability that it is either rock, paper, or scissors.

model = keras.models.Sequential([
    #First convolution
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    #Second Covolution
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    #Third Convolution
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #Fourth convolution
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #Flatten to go into deep neural network
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    # 512 neuron layer
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(numClasses, activation='softmax')
])

#Compiles the model, selects the adam optimize to update the model based on loss, measures the loss (discrepency from expected), and the metrics measure the accuarcy of the test
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#Trains the model using training images by going through (epochs=10) times
model.fit(trainImages, trainLabels, epochs=10, batch_size=32)

#Prints the test accuracy and testloss after evaluating the testImages
test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)
print("Test accuracy:", test_acc)

# newImage = Image that you want to classify as rock, paper, or scissors
newImage = cv2.imread("paper-test-image.png")
newImage = preprocessImage(newImage)
newImage = np.expand_dims(newImage, axis=0) #Expands the image to match expected shape of model

prediction = model.predict(newImage)
predictedSymbol = np.argmax(prediction) #returns the value of rock, paper, or scissors

#Prints out predicted result
if (predictedSymbol == 0):
    print("Predicted Symbol: Paper")
elif (predictedSymbol == 1):
    print ("Predicted Symbol: Rock")
elif(predictedSymbol == 2):
    print ("Predicted Symbol: Scissors")
