import librosa
import statsmodels.api as sm
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import pickle
import cv2
import mediapipe as mp
from cv2 import VideoCapture
from tensorflow import keras
import threading
import time
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def init_video_model():
    # Load the video model
    model_path = os.path.join(os.getcwd(), "video_model_v2.keras")
    video_model = keras.models.load_model(model_path)

    return video_model


@st.cache_resource
def init_transcribe_model():
    # Speech to text model
    transcribe_model = whisper.load_model("base")

    return transcribe_model


@st.cache_resource
def init_llm():
    model_path = r"C:\Users\aakan\AppData\Local\nomic.ai\GPT4All\mistral-7b-instruct-v0.1.Q4_0.gguf"   #orca-mini-3b-gguf2-q4_0.gguf
    llm = GPT4All(model=model_path, n_threads=8)

    return llm


video_model = init_video_model()
transcribe_model = init_transcribe_model()
llm = init_llm()


# path for saving mp3 audio file
audio_file_path = os.path.join(os.getcwd(), "uploaded_data", "audio")
mp3_file = audio_file_path + "/" + "audio.mp3"
if not os.path.exists(audio_file_path):
    os.makedirs(audio_file_path)

# path for saving mp3 audio chunks
audio_chunks_path = os.path.join(os.getcwd(), "uploaded_data", "audio_chunks")
if not os.path.exists(audio_chunks_path):
    os.makedirs(audio_chunks_path)

# Load actions.pkl file
action_main_path = os.path.join(os.getcwd(), "actions.pkl")
with open(action_main_path, 'rb') as file:
    actions = pickle.load(file)

# Media pipe Initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# global vars - multithreading
audio_list = []
video_list = []
txt_list = []


def extract_audio_from_mp4(video, mp3_file):
    # Load the video clip
    video_clip = VideoFileClip(video)
    # Extract the audio from the video clip
    audio_clip = video_clip.audio
    # Write the audio to a separate file
    audio_clip.write_audiofile(mp3_file)
    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    return None


def transcribe(mp3_file):
    result = transcribe_model.transcribe(mp3_file)
    txt_list.append(result["text"])

    return None


def find_pitch(data, sampling_frequency):
    auto = sm.tsa.acf(data)
    try:
        peaks = find_peaks(auto)[0] # Find peaks of the autocorrelation
        lag = peaks[0] # Choose the first peak as our pitch component lag
        pitch = sampling_frequency / lag # Transform lag into frequency
    except:
        pitch = 0

    return pitch


def find_root_mean_square_energy(data, hop_length=256, frame_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse = rmse[0]
    mean_rmse = np.sum(rmse)

    return mean_rmse


def calculate_pitch_energy(audio_chunks_path):
    audio_file_list = glob.glob(f"{audio_chunks_path}/*.mp3")

    pitch_energy = []
    for audio_chunk in audio_file_list:
        data, sampling_frequency = librosa.load(audio_chunk)

        pitch = find_pitch(data, sampling_frequency)
        energy = find_root_mean_square_energy(data)
        pitch_energy.append(round(pitch/energy))

    return pitch_energy


def get_audio_in_equal_chunks(mp3_file, audio_chunks_path):
    # Load your audio file
    audio = AudioSegment.from_file(mp3_file)
    # Calculate the length of each chunk
    chunk_length = len(audio) // 3
    # Split the audio into three chunks
    chunk1 = audio[:chunk_length]
    chunk2 = audio[chunk_length:2*chunk_length]
    chunk3 = audio[2*chunk_length:]
    # Export the chunks as separate files
    chunk1.export(audio_chunks_path + "/" + "chunk1.mp3", format="mp3")
    chunk2.export(audio_chunks_path + "/" + "chunk2.mp3", format="mp3")
    chunk3.export(audio_chunks_path + "/" + "chunk3.mp3", format="mp3")

    return None


def get_pitch_energy_label(pitch_energy):
    if (pitch_energy[0] <= pitch_energy[1]) and (pitch_energy[1] >= pitch_energy[2]):
        return "The pitch/energy rises and then falls"
    elif (pitch_energy[0] >= pitch_energy[1]) and (pitch_energy[1] <= pitch_energy[2]):
        return "The pitch/energy falls and then rises"
    elif (pitch_energy[0] == pitch_energy[1]) and (pitch_energy[1] == pitch_energy[2]):
        return "The pitch/energy does not change"
    elif (pitch_energy[0] >= pitch_energy[1]) and (pitch_energy[1] >= pitch_energy[2]):
        return "The pitch/energy decreases from high to low"
    elif (pitch_energy[0] <= pitch_energy[1]) and (pitch_energy[1] <= pitch_energy[2]):
        return "The pitch/energy increases from low to high"


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    return np.concatenate([pose, face])


def audio_main(mp3_file, audio_chunks_path):
    get_audio_in_equal_chunks(mp3_file, audio_chunks_path)
    pitch_energy = calculate_pitch_energy(audio_chunks_path)
    pitch_energy_label = get_pitch_energy_label(pitch_energy)
    audio_list.append(pitch_energy_label)

    return None


def video_main(video):
    capture = VideoCapture(video)

    sequence = []
    sentence = []
    threshold = 0.50

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while capture.isOpened():
            ret, frame = capture.read()
            # If an image frame has been grabbed, display it
            if ret == True:
                # Make holistic pose detections
                _, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = video_model.predict(np.expand_dims(sequence, axis=0))[0]

                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

            else:
                break

        capture.release()

    for sent in sentence:
        video_list.append(sent)

    return None


def llm_prompt(para):

    examples = [
        {
            "question": "In the speech, The pitch/energy raises then falls. The speaker is angry. The content of the speech is: I like sushi",
            "answer": """
            Sentiment: Negative
            Explanation: Though content of the speech is positive but speaker is angry hence overall sentiment is negative
            """,
        },
        {
            "question": "In the speech, The pitch/energy increases from high to low. The speaker is happy. The content of the speech is: I like sushi",
            "answer": """
            Sentiment: Positive
            Explanation: The content of the speech is positive also speaker is happy hence overall sentiment is positive
            """,
        },
        {
            "question": "In the speech, The pitch/energy raises then falls. The speaker is angry. The speaker is neutral. The content of the speech is: What a movie!",
            "answer": """
            Sentiment: Negative
            Explanation: Though content of the speech is positive but speaker is neutral and angry hence overall sentiment is negative
            """,
        },
        {
            "question": "In the speech, The pitch/energy raises then falls. The speaker is neutral. The content of the speech is: weather update. What a weather!",
            "answer": """
            Sentiment: Positive
            Explanation: The content of the speech is positive also speaker is neutral hence overall sentiment is positive
            """,
        },
    ]

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template="Question: {question}\n{answer}"
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    chain_response = chain.run(para)

    return chain_response


def get_prompt_temp(audio_list, video_list, txt_list):
    video_txt = []
    video_set = set(video_list)

    for lbl in video_set:
        video_txt.append(f"The speaker is {lbl}. ")

    txt_frmt = f"The content of the speech is: {txt_list[0]}"

    vid_prmpt = ''.join(video_txt)
    prmpt = f"In the speech, {audio_list[0]}. {vid_prmpt}{txt_frmt}"

    return prmpt


def main():
    st.title("Multimodal Sentiment Analysis")

    # Specify the folder to save the uploaded files
    save_folder = os.path.join(os.getcwd(), 'uploaded_data', 'video')

    # Create the folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to the specified folder
        video = os.path.join(save_folder, uploaded_file.name)
        with open(video, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # display the video file
        st.video(video)

    if st.button("Submit"):
        start_time = time.perf_counter()

        extract_audio_from_mp4(video, mp3_file)

        t1 = threading.Thread(target=video_main, args=(video,))
        t2 = threading.Thread(target=audio_main, args=(mp3_file, audio_chunks_path,))
        t3 = threading.Thread(target=transcribe, args=(mp3_file,))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        prmpt = get_prompt_temp(audio_list, video_list, txt_list)
        response = llm_prompt(prmpt)

        st.write(f"**Input Prompt:** {prmpt}")
        st.write(f"**Predicted Sentiment of the given video:** {response}")

        end_time = time.perf_counter()

        print(f'Time taken: {end_time - start_time}')


if __name__ == "__main__":
    main()
