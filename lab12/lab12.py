import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import utils
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import gdown
import cv2


# The following lines allows to exploit the GPU and make the training faster (Check in the Runtime - Change Runtime time if the GPU is set)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

# Loading the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Visualizing data's shape -> (50000, 32, 32, 3): 50000 images, 32x32 pixels, 3 channels (RGB)
print(x_train.data.shape)

# print(x_train[0][0][0])
# print(y_train)

# QUESTION 1(b) Plot some samples to visualize example of data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def plot_sample(x_train, y_train):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[i].astype("uint8"))
        plt.title(class_names[int(y_train[i])])
        plt.axis("off")

    plt.show()

# plot_sample(x_train, y_train)

# QUESTION 1(c) Preprocess the data
# Dividing for 255.0 each value in the whole np.array which contains pixels each comprised of three values (R, G, B) in a range 0-255 
# The result is the same np.array scaled down to 0-1
def pre_process_data(x_train, x_test, verbose=False):
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Checking that scaling worked
    print('Min in training set:', x_train.min())
    print('Max in training set:', x_train.max())

    print('Min in test set:', x_test.min())
    print('Max in test set:', x_test.max())

    return x_test, x_train


# x_test, x_train = pre_process_data(x_train, x_test)


def build_model():
    # Building the network
    model = models.Sequential()

    # First layer, Conv2D with 32 filters, kernel size 3x3, activation function relu
    # Conv2D output shape includes some padding, so it will be (30, 30, 32) (inputs were 32x32 pixels, 2 get left out by the 3x3 kernel)
    # Output size=(Input size−Filter size​+2*Padding)/Stride + 1 = ((32, 32) - (3, 3) + 2 * 0) / 1 + 1 = ((29, 29)) + 1 = (30, 30)
    # The first None simply means that the batch size isn't fixed, while the last 32 is related to the number of filters (kernels)
    # Param #:  32 filters, 3x3 kernel, input shape (32, 32, 3):
    # (filter_height * filter_width * input_channels + 1) * number_of_filters -> [(3 * 3 * 3 + 1) * 32 = (27 + 1) * 32 = 28 * 32 = 896]
    model.add(layers.Conv2D(32, (3, 3), 1, activation="relu", input_shape=(32, 32, 3)))
    # Second layer, MaxPooling2D with kernel size 2x2
    # Output size here will be half of the original (in terms of pixels, not filters) -> (15, 15, 32)
    model.add(layers.MaxPooling2D((2, 2)))
    # Third layer, Conv2D with 64 filters, kernel size 3x3, activation function relu
    # Output size=(Input size−Filter size​+2*Padding)/Stride + 1 = ((30, 30) - (3, 3) + 2 * 0) / 1 + 1 = ((27, 27)) + 1 = (28, 28)
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # Fourth layer, MaxPooling2D with kernel size 2x2
    # Like second layer -> (13, 13) / 2 = (6, 6)
    model.add(layers.MaxPooling2D((2, 2)))
    # Fifth layer, Conv2D with 64 filters, kernel size 3x3, activation function relu
    # Output size=(Input size−Filter size​+2*Padding)/Stride + 1 = ((6, 6) - (3, 3) + 2 * 0) / 1 + 1 = ((3, 3)) + 1 = (4, 4)
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # Sixth layer, flatten layer (3D to 1D)
    # Flattening from three dimensions to one -> (4, 4, 64) -> 4 * 4 * 64 = 16 * 64 = 1024
    model.add(layers.Flatten())
    # Seventh layer, Dense layer with 64 neurons and activation function relu
    # Output size: 64 neurons means 64 outputs
    model.add(layers.Dense(64, activation="relu"))
    # Eight and final layer, flatten layer for returning 10 classes
    # Output size: flattened to 10
    model.add(layers.Dense(10))

    model.summary()
    # keras.utils.plot_model(model, show_shapes=True)

    # QUESTION 1(c) - PartB Compile the CNN model according to the specifications
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model


def fit_model(model):
    model = build_model()

    #QUESTION 1(d): Train and test on the CIFAR10 dataset with 10 epochs to check the performance
    history = model.fit(x_train, y_train, epochs=10)

    model.save('my_model.h5') 

    return history


# model = build_model()
# fit_model(model)
model = models.load_model("my_model.h5")

def evaluate_model(model):
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# evaluate_model()

#QUESTION 1(e): Compute the confusion matrix
def confusion_matrix(y_test, y_predicted):
    # Test model
    y_predicted = model.predict(x_test)
    y_predicted = np.argmax(y_predicted, axis=1)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print("\n")
    plt.show()

    for c in range(10):
        x = np.where(y_test == c)
        print(c, ' - Number of samples of class ', class_names[c], ' is: ', len(x[0]))


# confusion_matrix(y_test, y_predicted)

# QUESTION 1(f):Test your model on new images about the available class that you can download from the web
def predict_from_urls(images_url, images_label):
    for u in range(len(images_url)):
        url = images_url[u]
        output = "./content/img.jpg"  # Output path for the downloaded image from Google drive

        # Download and read the image
        gdown.download(url, output, quiet=True)
        img = cv2.imread(output, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to adjust the color

        # Resize the image according to the dimension accepted by the model
        # Tensor to resize the image
        resize = tf.image.resize(rgb_img, (32,32)) # See before the CNN works with 32x32 as image size

        # Show resized image which is then fed to the model
        # plt.imshow(resize.numpy().astype(int))
        # plt.title(class_names[images_label[u]])
        # plt.show()

        # Predictions
        pred = model.predict(np.expand_dims(resize/255, 0))
        probabilities = tf.nn.softmax(pred[0]).numpy()
        print("\nPREDICTED LABEL: " + class_names[np.argmax(pred)])
        print("Probabilities:")
        for class_name, probability in zip(class_names, probabilities):
            print(f"{class_name}: {(probability * 100):.2f}%")

        # Show the image
        plt.imshow(rgb_img)
        plt.title(f"Actual: {class_names[images_label[u]]} - Predicted: {class_names[np.argmax(pred)]}")
        plt.show()


def predict_from_file(path, actual_label):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to adjust the color

    # Resize the image according to the dimension accepted by the model
    # Tensor to resize the image
    resize = tf.image.resize(rgb_img, (32,32)) # See before the CNN works with 32x32 as image size

    # Show resized image which is then fed to the model
    # plt.imshow(resize.numpy().astype(int))
    # plt.title(class_names[images_label[u]])
    # plt.show()

    # Predictions
    pred = model.predict(np.expand_dims(resize/255, 0))
    probabilities = tf.nn.softmax(pred[0]).numpy()
    print("\nPREDICTED LABEL: " + class_names[np.argmax(pred)])
    print("Probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"{class_name}: {(probability * 100):.2f}%")

    # Show the image
    plt.imshow(rgb_img)
    plt.title(f"Actual: {actual_label} - Predicted: {class_names[np.argmax(pred)]}")
    plt.show()


# images_url = ["https://drive.google.com/uc?id=1WTXIeWjoXwDgw2NGygAe11TRHpz4Y0cT",
#               "https://drive.google.com/uc?id=1GlYCU3N4WqqIrIcC9V_1EtjNOYijWUsW",
#               "https://drive.google.com/uc?id=14yYM5lYd6xug6_P3y3x96qeN0I7e5lsw",
#               "https://drive.google.com/uc?id=1eWSdGzzB089nXwaj9BLurPXKTHK2mULg",
#               "https://drive.google.com/uc?id=1ato5-H1uJm1ET95hIdnh6O8F3kplxkEy",
#               "https://drive.google.com/uc?id=1B6ocPfCNxTX5aq8CCM196NuOFgxhX-if",
#               "https://drive.google.com/uc?id=1Q73FXbnG5EiKMNkjnfzbCUjBeVtzlz3i",
#               "https://drive.google.com/uc?id=11EKbn0jlXNvHmXMRHLrOA0xgQq7a_Gph",
#               "https://drive.google.com/uc?id=1xVWmXw_BWIGpphw055JovE78ZQGrDYyZ"]

images_url = ["https://www.purina.it/sites/default/files/2021-02/BREED%20Hero_0034_chihuahua_smooth.jpg",
            "https://www.milanotoday.it/~media/horizontal-hi/18288975154480/l-aere-mundial-a-milano-foto-da-tw-volandia3.jpg"]

images_label = [class_names.index('dog'), class_names.index('airplane')]

predict_from_urls(images_url, images_label)
# predict_from_file("./content/img.jpg", "dog")