# Building a Flask app on Image Classification of Melonoma Dataset implemented by Convolutional Neural Network (CNN)

This is the project that finished after the 2nd week of studying **Machine Learning**.





## INTRODUCTION
### 1. Melnoma Dataset
**Melnoma** [dataset](https://www.kaggle.com/amyjang/tensorflow-transfer-learning-melanoma/dataa) provided by Kaggle with 116 GB data of Melnoma images with the labels 

### 2. Project goals
- Building a **deep neural network** using **TensorFlow** to classify melonoma images.

- Making a **Flask application** so user can upload their photos and receive the prediction.

### 3. Project plan

During this project, we need to answer these following questions:

**A. Build the model**
- How to import the data
- How to preprocess the images
- How to create a model
- How to train the model with the data
- How to export the model
- How to import the model
    
**B. Build the Flask app**

**Front end**
- HTML
    - How to connect frontend to backend
    - How to draw a number on HTML
    - How to make UI looks good

**Back end**
- Flask
    - How to set up Flask
    - How to handle backend error
    - How to make real-time prediction
    - Combine the model with the app


## SETUP ENVIRONMENT
* In order to run our model on a Flask application locally, you need to clone this repository and then set up the environment by these following commands:

```shell
python3 -m pip install --user pipx
python3 -m pipx ensurepath

pipx install pipenv

# Install dependencies
pipenv install --dev

# Setup pre-commit and pre-push hooks
pip install pre-commit
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```
* On the Terminal, use these commands:
```
# enter the environment
pipevn shell
pipenv graph
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=1
flask run
```
* If you have error `ModuleNotFoundError: No module named 'tensorflow'` then use
```
pipenv install tensorflow==2.0.0beta-1
```
* If  `* Debug mode: off` then use
```
export FLASK_DEBUG=1
```

* Run the model by 

```shell
pipenv run flask run
```

* If you want to exit `pipenv shell`, use `exit`

## HOW IT WORK: CONVOLUTIONAL NEURAL NETWORK (CNN)

> In deep learning, a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. (Wiki)

For this project, we used **pre-trained model [MobileNetV2](https://keras.io/applications/#mobilenetv2)** from keras. MobileNetV2 is a model that was trained on a large dataset to solve a **similar problem to this project**, so it will help us to save lots of time on buiding low-level layers and focus on the application.

***Note:** You can learn more about CNN architecture [here](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)*  

![](https://www.datascience.com/hs-fs/hubfs/CNN%202.png?width=650&name=CNN%202.png)

### 1. Load and preprocess images:

- Import **path, listdir** from **os library**.
- Find and save all the image's path to **all_image_paths**. (note that our images is in folder `train`). 

```python 
all_image_path = [path.join('train', p) for p in listdir('train') if isfile(path.join('train', p))]
```

- Define a function to load and preprocess image from path:

```python
def load_and_preprocess_image(path):
    file = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file , channels=3)
    image = tf.image.resize(image, [192, 192]) # resize all images to the same size.
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range
    return image
```

- Load and preprocess all images which path is in **all_image_path**:

```python
all_images = [load_and_preprocess_image(path) for path in all_image_path]
```
- Save all image labels in **all_image_labels**:

```python

labels = [path.split('.')[0][-3:] for path in all_image_path] 

# Transfer name-labels to number-labels:
all_image_labels = [dict[label] for label in labels]
```

- To implement batch training, we put the images and labels into Tensorflow dataset:

```python
ds = tf.data.Datasets.from_tensor_slices((all_images, all_image_labels))
```

### 2. Building CNN model: 

The CNN model contain **MobileNetV2, Pooling, fully-connected hidden layer and Output layer**.

- First we create **mobile_net** as an instance of **MobileNetV2**:

```python
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False # this told the model not to train the mobile_net.
```

- Then we build CNN model:

```python
cnn_model = keras.models.Sequential([
    mobile_net, # mobile_net is low-level layers
    keras.layers.GlobalAveragePooling2D(), 
    keras.layers.Flatten(), 
    keras.layers.Dense(64, activation="relu"), # fully-connected hidden layer 
    keras.layers.Dense(2, activation="softmax") # output layer
])
```

### 3. Training model:

We almost there! But before training our **cnn_model**, we need to implement batch to the training data so that the model will train faster.

```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = ds.shuffle(buffer_size = len(all_image_labels))
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```

Now we train the model.

```python
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
```

```python
steps_per_epoch=tf.math.ceil(len(all_image_dirs)/BATCH_SIZE).numpy()
cnn_model.fit(train_ds, epochs=2, steps_per_epoch=steps_per_epoch)
```
After training, save the model for later use.
```python
cnn_model.save('my_model.h5')
```

## MODEL PERFOMANCE SUMARY
Our model has the accuracy of **97.79 %** for the train dataset and **97.32 %** for the test dataset. 

## FLASH APPLICATION
# Melanoma Image Classification with Flask Application

## Project Page


## Flask application

While I had been exploring implementation through Flutter for app deployment, Flask seemed much more feasible given my time constraints and level of expertise.

## Home page:

<img src="images/Homepage.png">

The homepage asks for the user to upload a JPEG of any size into the application and to press SUBMIT once done.


## Results page:

<img src="images/Results.png">

Upon pressing SUBMIT, you automatically get transferred to the Results page, and you are given a message to get the mole checked out or that it is just another beauty mark.  The confidence level of that prediction is also given.
   




## References:

International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration [https://doi.org/10.34970/2020-ds01][4] (2020).

Rotemberg, V. _et al_. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. _Sci. Data_ 8: 34 (2021). [https://doi.org/10.1038/s41597-021-00815-z][5]

ISIC 2019 data is provided courtesy of the following sources:

- BCN20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
- HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [https://doi.org/10.1038/sdata.2018.161][6]
- MSK Dataset: (c) Anonymous; [https://arxiv.org/abs/1710.05006][7] ; [https://arxiv.org/abs/1902.03368][8]

Tschandl, P. _et al_. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. _Sci. Data_ 5: 180161 doi: 10.1038/sdata.2018.161 (2018)

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.

Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: “BCN20000: Dermoscopic Lesions in the Wild”, 2019; arXiv:1908.02288.

Codella, N. _et al_. “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; [https://arxiv.org/abs/1902.03368][9]


[4]:	https://doi.org/10.34970/2020-ds01
[5]:    https://doi.org/10.1038/s41597-021-00815-z
[6]:	https://doi.org/10.1038/sdata.2018.161
[7]:	https://arxiv.org/abs/1710.05006
[8]:	https://arxiv.org/abs/1902.03368
[9]:	https://arxiv.org/abs/1902.03368

## CONCLUSION

We successfully **built a deep neural network model** by implementing **Convolutional Neural Network (CNN)** to classify Melonoma images with very high accuracy **97.32 %**.
In addition, we also **built a Flask application** so user can upload their images and classify easily.
