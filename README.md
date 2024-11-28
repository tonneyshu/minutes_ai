# README

## Overview

![image](https://github.com/user-attachments/assets/5e74fe85-f888-4d45-b37e-f1ac9f3e0614)

**AI Minutes** is a Streamlit application that leverages OpenAI's Whisper (large-v3) ASR model to transcribe audio files, pyannote's speaker-diarization-3.1 on Hugging Face for speaker diarization and summarize the transcriptions using LangChain's summarization chain on llama-3.1-70b-versatile (Groq API).

## Features

- **Audio Transcription**: Upload audio files in MP3 or WAV format, and the application will transcribe the audio to text.
- **Speaker Diarization**: The application can identify multiple speakers in the audio and label their contributions accurately.
- **Text Summarization**: After transcription, it summarizes the conversation, highlighting key points, decisions made, and action items.

## Requirements

Ensure you have Python 3.10 or higher installed. All required packages are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/tonneyshu/minutes_ai.git
   ```

2. **Environment Variables**: Create a `.env` file in the root directory of your project and add your Hugging Face token:
   ```plaintext
   HF_TOKEN=<your_hugging_face_token>
   ```

3. **Run the Application**:
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Upload an audio file (MP3 or WAV).
3. Specify the number of speakers in the audio.
4. Wait for the transcription and summarization processes to complete.
5. View the transcription and summary displayed on the page.

## Code Explanation

### Key Functions

- **load_whisper_model()**: Loads the WhisperX model for transcription.
- **transcribe_audio(audio_file, model, num_speakers)**: Handles audio file processing and returns a transcribed text with speaker labels.
- **fix_speaker_label(text)**: Reviews and corrects any mis-labeled speaker instances in the transcription.
- **summarize_text(text)**: Summarizes the conversation text, identifying key points and action items.

### GPU Memory Management

The application includes functions to manage GPU memory effectively, ensuring that resources are freed after use to prevent memory leaks.

### Caching

Streamlit's caching mechanism is used to optimize model loading times, ensuring that models are loaded only once during a session.

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX)
