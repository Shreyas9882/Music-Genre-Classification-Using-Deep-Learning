

# Music Genre Classification Using Deep Learning

## 1. Introduction to Deep Learning

Deep Learning is a subset of machine learning that uses neural networks with many layers to learn from vast amounts of data. It is particularly effective for tasks such as image recognition, speech analysis, and audio classification. In this project, we utilize **Convolutional Neural Networks (CNNs)**, a type of deep learning model, to classify music genres. The audio data is transformed into **Mel spectrograms**, which are visual representations of sound, allowing the CNN to learn patterns and predict the genre of the music.


## 2. Introduction to the Project

This **Music Genre Classification System** uses deep learning techniques to automatically classify music tracks into 10 predefined genres. The system takes an audio file as input, preprocesses it by converting the audio into Mel spectrograms, and then uses a pre-trained deep learning model to predict the genre of the music.

The goal of this project is to demonstrate the power of AI in music analysis and provide a tool that can classify music genres accurately and efficiently. The application is built using **Streamlit** for the user interface and **TensorFlow** for deep learning.

### List of Genres
The model is trained to classify music into the following genres:
- **Blues**
- **Classical**
- **Country**
- **Disco**
- **Hip-Hop**
- **Jazz**
- **Metal**
- **Pop**
- **Reggae**
- **Rock**


## 3. Running the Project

### Steps to Run the Application:
1. **Clone the Repository:**
   - Start by cloning the project repository from GitHub:
   ```bash
   git clone <repository_url>
   ```

2. **Install Dependencies:**
   - Install the required libraries using pip:
   ```bash
   pip install tensorflow streamlit librosa matplotlib numpy
   ```

3. **Run the Streamlit Application:**
   - Navigate to the project directory and run the app:
   ```bash
   streamlit run Music_Genre_App.py
   ```
   - Once the server starts, open the displayed URL in your browser to access the Music Genre Classification System.

### Features of the Application:
- **Home Page**: An introduction to the system, instructions, and an overview.
- **About Project**: Information about the dataset, genres, and the model used.
- **Prediction Page**: Upload your audio file and get the predicted genre.

### How It Works:
1. **Upload Audio**: The user uploads an MP3 file on the prediction page.
2. **Preprocessing**: The system breaks the audio file into smaller chunks and converts each chunk into a Mel spectrogram, a visual representation of the audio.
3. **Model Prediction**: The preprocessed Mel spectrogram data is fed into a pre-trained deep learning model (CNN) to predict the genre of the music.
4. **Display Results**: The predicted genre is displayed on the app, and the user can play the audio.

<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/897a8c4c-6204-49e2-83e3-54e65eb6b828" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/84146263-41e8-4b17-8f8e-f88329a7cd6b" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/eec65b4e-e170-4618-85d6-e404bc89946f" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/5c9565da-cb22-422f-a2ef-11ba7dbd65e8" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/dbfb04cc-5be7-4fc6-aa5a-9d3e95cd7f13" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/9864d927-1b68-4339-8338-e767ecff2731" />
<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/4d7c20f1-7480-4d15-a66a-01c7bac6b373" />

## 4. Prerequisites for the Project

Before running the application, make sure to install the following dependencies:

```bash
pip install tensorflow
pip install streamlit
pip install librosa
pip install matplotlib
pip install numpy
```

### Pre-trained Model

The system uses a pre-trained deep learning model (`Trained_model.h5`). Make sure to download this model and place it in the project directory. The model is a **CNN** trained on the **GTZAN** dataset, which contains audio files across 10 music genres.

