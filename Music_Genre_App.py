import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
from tensorflow.image import resize


# Function
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


# Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]


# Sidebar
st.sidebar.title("🌟 Music Genre Classification Dashboard 🌟")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

## Main Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e2f;  /* Dark background */
            color: white;
            font-family: 'Arial', sans-serif;
        }
        h2, h3, h4 {
            color: #f0f0f5;
        }
        .stImage {
            border-radius: 10px;
        }
        .stButton {
            background-color: #f88a00;
            color: white;
        }
        .stButton:hover {
            background-color: #e67e00;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(''' ## Welcome to the **Music Genre Classification System**! 🎶🎧''')

    # Add a more vibrant image
    image_path = "MUSIC-GENRES.png"
    st.image(image_path, use_column_width=True, caption="Discover Your Music's Genre!")

    st.markdown("""
    **Our goal is to help in identifying music genres from audio tracks efficiently.**  
    Upload an audio file, and our system will analyze it to detect its genre.  
    Discover the power of AI in music analysis!

    ### How It Works
    1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
    2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
    3. **Results:** View the predicted genre along with related information.

    ### Why Choose Us?
    - **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
    - **User-Friendly:** Simple and intuitive interface for a smooth user experience.
    - **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

    ### Get Started
    Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

    ### About Us
    Learn more about the project, our team, and our mission on the **About** page.
    """)

# About Project Page
elif app_mode == "About Project":
    st.markdown("""
    ### About the Project
    Music has intrigued experts for a long time. What differentiates one song from another?  
    How can we visualize sound? What makes a tone different from another? This project seeks to answer such questions.

    ### About the Dataset
    #### Content
    1. **Genres Original**: A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds).
    2. **List of Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock.
    3. **Images Original**: A visual representation for each audio file. Audio files are converted to Mel Spectrograms to be processed by neural networks.
    4. **CSV Files**: One file with mean and variance computed over multiple audio features for each song, and another file with 3-second audio splits to increase data volume.

    ### Why This Project Matters
    By leveraging AI and deep learning, we can automate the classification of audio into genres, enabling faster music categorization and analysis.
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("🎧 **Model Prediction**")

    test_mp3 = st.file_uploader("Upload an audio file (.mp3)", type=["mp3"])

    if test_mp3 is not None:
        filepath = 'Test_Music/' + test_mp3.name

    # Audio Play Button with nice style
    if st.button("🔊 Play Audio"):
        st.audio(test_mp3)

    # Prediction Button
    if st.button("🔮 Predict Genre"):
        with st.spinner("Analyzing... Please wait!"):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            st.balloons()

            # Enhanced Balloon Effect and Prediction Message
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

            # Displaying result with beautiful fonts and styling
            st.markdown("""
            <style>
            .prediction {
                font-size: 28px;
                font-weight: bold;
                color: #f88a00;
                text-align: center;
                text-shadow: 3px 3px 8px rgba(0,0,0,0.5);
            }
            .balloon {
                background: url('https://img.icons8.com/ios/452/balloon.png') no-repeat center center;
                background-size: contain;
                width: 100px;
                height: 100px;
                animation: floatBalloon 2s ease-in-out infinite;
            }
            @keyframes floatBalloon {
                0% { transform: translateY(0); }
                50% { transform: translateY(-30px); }
                100% { transform: translateY(0); }
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="prediction">🎵 It\'s a {label[result_index].capitalize()} music! 🎶</div>',
                unsafe_allow_html=True)

            # Display the floating balloon
            st.markdown('<div class="balloon"></div>', unsafe_allow_html=True)

            # You can add more floating balloons or animations here for extra effect
