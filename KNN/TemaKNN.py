import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
import seaborn as sns
import pandas as pd
import cv2


# load the image
def load_image(path):
    image = Image.open(path)

    # convert image to numpy array
    data = asarray(image)
    
    return data.flatten()


# plot digits
def plot_digits(data, row_size):
    for i in range(len(data)):
        ax = plt.subplot(row_size, row_size, i+1)
        
        digit = data[i]
        digit_image = digit.reshape(28, 28)
        
        plt.imshow(digit_image, cmap=plt.cm.Blues)
        plt.axis("off")


# load train dataset
X_train = [] # data
Y_train = [] # labels

for i in range(0, 10):
    files = [f for f in os.listdir("./train/"+str(i)+"/") if os.path.isfile("./train/"+str(i)+"/"+f)]
    for f in files:
        X_train.append(load_image("./train/"+str(i)+"/"+f))
        Y_train.append(i)

X_train = np.stack(X_train, axis=0)
X_train = X_train / 255 # scale data

Y_train = np.array(Y_train)


plot_digits(X_train[:25], row_size=5)
plt.show()


# load test dataset 
X_test = [] # data
Y_test = [] # labels

for i in range(0, 10):
    files = [f for f in os.listdir("./test/"+str(i)+"/") if os.path.isfile("./test/"+str(i)+"/"+f)]
    for f in files:
        X_test.append(load_image("./test/"+str(i)+"/"+f))
        Y_test.append(i)

X_test = np.stack(X_test, axis=0)
X_test = X_test / 255 # scale data

Y_test = np.array(Y_test)


# load validation dataset
X_validation = [] # data
Y_validation = [] # labels

for i in range(0, 10):
    files = [f for f in os.listdir("./validation/"+str(i)+"/") if os.path.isfile("./validation/"+str(i)+"/"+f)]
    for f in files:
        X_validation.append(load_image("./validation/"+str(i)+"/"+f))
        Y_validation.append(i)

X_validation = np.stack(X_validation, axis=0)
X_validation = X_validation / 255 # scale data

Y_validation = np.array(Y_validation)


# K-Nearest Neighbors - KNN
kValues = range(1, 11)
accuracies = []

start_time = datetime.now()
for k in range(1, 11):
    # train the k-Nearest Neighbor classifier
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, Y_train)

    # evaluate the model on validation dataset
    scoreValidation = model.score(X_validation, Y_validation)

    # validation results
    print("k = %d, accuracy = %.2f%%" % (k, scoreValidation * 100))
    accuracies.append(scoreValidation)

end_time = datetime.now()

# confusion matrix
predictions_validation = model.predict(X_validation)
print ("Confusion matrix - Validation")
confusionMatrix = confusion_matrix(Y_validation, predictions_validation)
print(confusionMatrix)

plt.figure(figsize= (9,9))
sns.heatmap(confusionMatrix, annot = True, fmt = ".3f")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Validation')
plt.show()


# training results
time_difference = (end_time - start_time).total_seconds()
print("Execution time of training is: ", time_difference, "seconds") 

# plot accuracies
plt.title('KNN')
plt.plot(kValues, accuracies, label = 'Validation Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# find the value of k that has the best accuracy
kMax = np.argmax(accuracies)
highestAccuracy = accuracies[kMax] * 100
print("k = %d : highest accuracy of %.2f%% on validation dataset" % (kValues[kMax], highestAccuracy))


# retrain the model on test dataset using kMax value
start_time = datetime.now() 
model = KNeighborsClassifier(n_neighbors = kValues[kMax])
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
end_time = datetime.now() 

# testing results
time_difference = (end_time - start_time).total_seconds()
print("Execution time of testing is: ", time_difference, "seconds") 


# accuracy for each digit
accuracies_per_digit = {digit: {"correct": 0, "total": 0} for digit in range(10)}

for digit in range(10):
    digit_indices = np.where(Y_test == digit)[0]
    predictions_digit = model.predict(X_test[digit_indices])
    
    correct_predictions = np.sum(predictions_digit == digit)
    total_samples = len(digit_indices)
    
    accuracies_per_digit[digit]["correct"] = correct_predictions
    accuracies_per_digit[digit]["total"] = total_samples
    
    accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
    print(f"Digit {digit}: Accuracy = {accuracy:.2f}%")

# overall accuracy
overall_accuracy = metrics.accuracy_score(Y_test, model.predict(X_test)) * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")


# classification report
print("Classification report")
report = classification_report(Y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

plt.figure(figsize=(10, 6))
plt.title('Classification Report')
plt.axis('off')  # Hide axes
plt.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center', 
    colColours=['lightgray']*len(report_df.columns), cellColours=[['lightgray']*len(report_df.columns)]*len(report_df.index), 
    bbox=[0, 0, 1, 1])
plt.show()

# confusion matrix
print ("Confusion matrix")
confusionMatrix = confusion_matrix(Y_test, predictions)
print(confusionMatrix)

plt.figure(figsize= (9,9))
sns.heatmap(confusionMatrix, annot = True, fmt = ".3f")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy score: {0}'.format(overall_accuracy))
plt.show()

# predicted digit vs actual digit
for i in np.random.randint(0, high=len(Y_test), size=(5,)):
    image = X_test[i]
    prediction = model.predict([image])[0]
         
    # show the prediction
    imgdata = np.array(image, dtype='float')
    pixels = imgdata.reshape((28,28))
    plt.imshow(pixels,cmap='gray')
    plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
    print("Predicted : {}".format(prediction))

    plt.show()
    cv2.waitKey(0)


# image path
image_path = "./testImage.png" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Redimensionează imaginea la dimensiunea 28x28
image = cv2.resize(image, (28, 28))

# load image
image = image / 255  # scale

# flatten image
image_vector = image.flatten()

# predict
predicted_digit = model.predict([image_vector])[0]

# print predicted digit
plt.imshow(image.reshape(28, 28), cmap=plt.cm.Blues)
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
