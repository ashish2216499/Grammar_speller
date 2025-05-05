import os
import speech_recognition as sr
import language_tool_python
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main audio folder
main_audio_folder = 'audios'
audio_data = []

# Initialize grammar tool once per process
def get_tool():
    return language_tool_python.LanguageTool('en-US')

def transcribe_audio(file_path):
    """
    Transcribes audio using Google's speech recognition.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        logging.warning(f"Could not understand the audio: {file_path}")
        return None
    except sr.RequestError as e:
        logging.error(f"API error for {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for {file_path}: {e}")
        return None

def check_grammar(text):
    """
    Corrects grammar in a given text.
    """
    try:
        tool = get_tool()
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception as e:
        logging.error(f"Grammar check failed: {e}")
        return text

def process_file(file_path, subfolder, filename):
    """
    Processes a single audio file.
    """
    logging.info(f"Processing file: {filename}")
    transcribed_text = transcribe_audio(file_path)
    if transcribed_text:
        corrected_text = check_grammar(transcribed_text)
        return (subfolder, filename, transcribed_text, corrected_text)
    logging.warning(f"Skipping {filename} as transcription failed.")
    return None

def process_subfolder(subfolder):
    """
    Processes all audio files in a subfolder.
    """
    folder_path = os.path.join(main_audio_folder, subfolder)
    tasks = []

    logging.info(f"Processing subfolder: {subfolder}")

    with ProcessPoolExecutor() as executor:
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                tasks.append(executor.submit(process_file, file_path, subfolder, filename))

        for task in tasks:
            result = task.result()
            if result:
                audio_data.append(result)

def main():
    logging.info("Starting the transcription and grammar correction process.")

    # Process files in both 'train' and 'test' subfolders
    for subfolder in ['train', 'test']:
        process_subfolder(subfolder)

    # Save the results to a CSV file
    if audio_data:
        df = pd.DataFrame(audio_data, columns=['Subfolder', 'Filename', 'Transcribed Text', 'Corrected Text'])
        df.to_csv('audio_transcriptions_and_grammar.csv', index=False)
        logging.info("Saved results to 'audio_transcriptions_and_grammar.csv'.")
    else:
        logging.warning("No transcriptions were successful.")

if __name__ == "__main__":
    main()
