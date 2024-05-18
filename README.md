## Audio Classification Project Description

### Dataset Used
The dataset used for this project is the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music. The dataset is divided into 10 folds to allow for cross-validation.

### Code Overview

#### 1. Audio Data Preprocessing

1. **Loading a Sample Audio using Librosa**:
   - The audio file is loaded using the `librosa` library. The audio data and sample rate are obtained.

2. **Plotting the Audio Data**:
   - The audio data is plotted to visualize the waveform. Librosa converts the signal to mono, meaning the channel will always be 1.

3. **Loading the Audio using Scipy**:
   - The audio file is also read using the `scipy` library to compare with the `librosa` output. The original audio with 2 channels is plotted.

#### 2. Feature Extraction

1. **Extracting MFCC Features**:
   - Mel-Frequency Cepstral Coefficients (MFCC) are used to summarize the frequency distribution across the window size. This allows analysis of both the frequency and time characteristics of the sound.

2. **Defining Feature Extraction Function**:
   - A function is defined to load an audio file and extract MFCC features. The features are scaled and averaged.

3. **Extracting Features for All Audio Files**:
   - The feature extraction function is applied to all audio files in the dataset. The extracted features are stored in a Pandas DataFrame.

#### 3. Data Preparation

1. **Splitting the Dataset**:
   - The features and class labels are extracted from the DataFrame and converted to numpy arrays.

2. **Label Encoding**:
   - The class labels are encoded using `LabelEncoder` and converted to categorical data for model training.

3. **Train-Test Split**:
   - The dataset is split into training and testing sets using an 80-20 split.

#### 4. Model Creation

1. **Building the Model**:
   - A Sequential neural network model is built using Keras. The model consists of multiple dense layers with dropout and ReLU activation functions.

2. **Compiling the Model**:
   - The model is compiled with categorical cross-entropy loss, accuracy metrics, and the Adam optimizer.

3. **Training the Model**:
   - The model is trained on the training data for 150 epochs with a batch size of 32. The best model is saved using ModelCheckpoint.

4. **Evaluating the Model**:
   - The model's accuracy is evaluated on the test set, achieving a test accuracy of 0.804.

#### 5. Testing with New Audio Data

1. **Preprocessing and Predicting New Audio Data**:
   - A new audio file is loaded and preprocessed in the same way as the training data. The MFCC features are extracted and scaled. The trained model predicts the class of the new audio data.

### Conclusion
This project involves loading and preprocessing audio data, extracting features using MFCC, building and training a neural network model to classify urban sounds, and testing the model with new audio data. The dataset used is the UrbanSound8K dataset, and the model achieved an accuracy of 0.804 on the test set.
