import re
from moviepy.editor import VideoFileClip, TextClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, clips_array, ImageClip,ImageSequenceClip
from moviepy.config import change_settings
import whisper_timestamped as whisper
import os
from moviepy.config import change_settings
import json
from pathlib import Path
from openai import OpenAI
import os
from pydub import AudioSegment
import psutil
import time
import datetime
import re
import spacy
from spacy import displacy
import math
import requests
from urllib.parse import urlparse
imagemagick_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ImageMagick'))

IMAGEMAGICK_BINARY = os.path.join(imagemagick_path, 'magick.exe')

change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

def GetAudio(mp3_location, wav_location):
    """
    Get audio clip and transcribe it using Whisper ASR.

    Args:
        mp3_location (str): The location of the MP3 audio file.
        wav_location (str): The location of the WAV audio file.

    Returns:
        tuple: A tuple containing the audio clip and the transcribed result.
    """
    print(mp3_location)
    print(wav_location)
    mp3_location_str = str(mp3_location)
    wav_location_str = str(wav_location)
    audio_clip = AudioFileClip(mp3_location_str)
    audio = whisper.load_audio(wav_location_str)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return audio_clip, result

def CalculateAudio(result, correct_period_indexes, position_threshold=20):

    """
    Calculate the subtitles, last end time, and words with punctuation for the given audio result.

    Args:
        result (dict): The audio result containing segments and words.
        correct_period_indexes (list): List of correct period indexes.
        position_threshold (int, optional): The position threshold for considering a word as the end of a sentence. Defaults to 20.

    Returns:
        tuple: A tuple containing the subtitles (list), last end time (float), and words with punctuation (list).
    """
      
    subtitles = []
    last_end_time = 0
    last_sentence_end_time = 0
    words_with_punctuation = []

    overall_char_index = 0  

    for segment in result['segments']:
        for word in segment['words']:
            start_time = word['start']
            end_time = word['end']
            word_text = word['text']

            subtitle_info = {
                "text": word_text,
                "start_time": start_time,
                "end_time": end_time
            }
            if last_end_time < end_time:
                last_end_time = end_time
            
            if word_text.endswith('.') and any(abs(overall_char_index - period_index) <= position_threshold for period_index in correct_period_indexes):
                if last_sentence_end_time == 0:
                    words_with_punctuation.append((word_text, end_time))
                    last_sentence_end_time = end_time
                else:
                    words_with_punctuation.append((word_text, end_time - last_sentence_end_time))
                    last_sentence_end_time = end_time

            subtitles.append(subtitle_info)
            overall_char_index += len(word_text) + 1

    return subtitles, last_end_time, words_with_punctuation

def GetVideoClip(video_length):

    """
    Retrieves a video clip with a duration closest to the specified video_length.

    Args:
        video_length (float): The desired duration of the video clip.

    Returns:
        tuple: A tuple containing the video clip and its height.

    Raises:
        None

    """

    video_length = round(video_length)
    video_length_str = str(video_length) + ".mp4"
    print(video_length_str)

    closest_duration_diff = float('inf')
    closest_video_path = None

    script_directory = os.path.dirname(os.path.abspath(__file__))
    videos_path = os.path.join(script_directory, 'videos')

    for root, dirs, files in os.walk(videos_path):
        print("root: " +root)
        print("dirs: " +str(dirs))
        print("files: " +str(files))
        for file in files:
            print("file: " +file)
            if file.endswith(".mp4"):
                print("files found")
                video_path = os.path.join(root, file)
                video_duration = VideoFileClip(video_path).duration

                duration_diff = abs(video_duration - video_length)
                print(duration_diff)
                
                if duration_diff < closest_duration_diff:
                    closest_duration_diff = duration_diff
                    closest_video_path = video_path

    if closest_video_path:
        print("Closest video:", closest_video_path)
        video_clip = VideoFileClip(closest_video_path)
    else:
        print("No video with similar duration found. Using default video.")
        video_clip = VideoFileClip("vid.mp4")

    video_clip = video_clip.without_audio()
    video_height = video_clip.size[1]
    return video_clip, video_height

def CreateSubtitles(subtitle_info,video_clip):

    """
    Create subtitle clips based on the provided subtitle information.

    Args:
        subtitle_info (list): A list of dictionaries containing subtitle information.
            Each dictionary should have the following keys:
            - "text" (str): The text of the subtitle.
            - "start_time" (str): The start time of the subtitle in seconds.
            - "end_time" (str): The end time of the subtitle in seconds.
        video_clip (VideoClip): The video clip to which the subtitles will be added.

    Returns:
        list: A list of TextClip objects representing the subtitle clips.
    """
      
    subtitle_clips=[]
    for subtitle in subtitle_info:

        text = subtitle["text"]
        start_time = subtitle["start_time"]
        end_time = subtitle["end_time"]
        
        start_time_seconds = float(start_time)
        end_time_seconds = float(end_time)

        duration = end_time_seconds - start_time_seconds

        # Create TextClip for the subtitle
        text_clip = TextClip(text, fontsize=24, color='white', bg_color='None', size=(video_clip.size[0], 50)).set_duration(duration).set_start(start_time_seconds)
        print( text_clip.duration)
        # Append the TextClip to the list
        subtitle_clips.append(text_clip)
    
    return subtitle_clips

def GenerateVideo(video_clip, subtitle_clips, audio_clip, video_height, random_code, output_path_name):
    """
    Generate a video with subtitles and audio.

    Args:
        video_clip (VideoClip): The main video clip.
        subtitle_clips (list): List of subtitle clips.
        audio_clip (AudioClip): The audio clip.
        video_height (int): The height of the video.
        random_code (str): Random code for the video name.
        output_path_name (str): Output directory path.

    Returns:
        None

    Raises:
        Exception: If there is an error saving the video.

    """
    script_directory = os.path.dirname(os.path.abspath(__file__))

    output_directory = os.path.join(script_directory, output_path_name)

    os.makedirs(output_directory, exist_ok=True)

    video_name = f"{random_code}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    video_path = os.path.join(output_directory, f"{video_name}.mp4")

    video_with_subtitles = CompositeVideoClip([video_clip.set_audio(audio_clip).set_duration(audio_clip.duration)] +[video_clip]
                                               + [subtitle_clips.set_pos(('center', video_height * 0.8)) for subtitle_clips in subtitle_clips])

    try:
        video_with_subtitles.write_videofile(video_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True)
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error saving video: {e}")

def SendAPIRequest(random_code,text_input):

    """
    Sends an API request to OpenAI's text-to-speech service and saves the generated audio file.

    Args:
        random_code (str): A random code used to generate unique audio file names.
        text_input (str): The input text to be converted into speech.

    Returns:
        tuple: A tuple containing the file paths of the generated audio files in MP3 and WAV formats.

    Raises:
        None

    Example:
        text_input = "Hello, world!"
        mp3_file, wav_file = SendAPIRequest(random_code, text_input)
        print(mp3_file)  # Output: "path/to/output/directory.file.mp3"
        print(wav_file)  # Output: "path/to/output/directory.file.wav"
        
    """
     
    #need to add your own api key to use openai text to speech
    key = os.environ.get("OPENAI_API_KEY")
    
    client = OpenAI(api_key=key)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy", #change voice here
        input=text_input
    )

    audios_folder = Path(__file__).parent / "Audios"

    audio_file_name_mp3 = f"audio_{random_code}.mp3"
    audio_file_name_wav = f"audio_{random_code}.wav"

    speech_file_path_mp3 = audios_folder / audio_file_name_mp3
    speech_file_path_wav = audios_folder / audio_file_name_wav

    response.stream_to_file(speech_file_path_mp3)

    audio = AudioSegment.from_mp3(speech_file_path_mp3)
    audio.export(speech_file_path_wav, format="wav")
    return(speech_file_path_mp3,speech_file_path_wav)

#test
def UseExistingAudio():
    audios_folder = Path(__file__).parent / "Audios"
    audio_file_name_mp3 = "audio_1f405de5-9be4-4c44-872f-8574e6695462.mp3"
    audio_file_name_wav = "audio_1f405de5-9be4-4c44-872f-8574e6695462.wav"
    speech_file_path_mp3 = audios_folder / audio_file_name_mp3
    speech_file_path_wav = audios_folder / audio_file_name_wav
    return(speech_file_path_mp3,speech_file_path_wav)
#test
def measure_resource_usage(start_cpu_percent, start_memory_info,start_time):

    end_time = time.time()

    end_cpu_percent = psutil.cpu_percent()
    end_memory_info = psutil.virtual_memory()

    elapsed_time = end_time - start_time
    cpu_usage_diff = end_cpu_percent - start_cpu_percent
    memory_usage_diff = end_memory_info.percent - start_memory_info.percent

    print(f'Total Elapsed Time: {elapsed_time:.2f} seconds')
    print(f'Total CPU Usage: {cpu_usage_diff:.2f}%')
    print(f'Total Memory Usage: {memory_usage_diff:.2f}%')

def GetMetadata():
    """
    Retrieves the path of the metadata JSON file from the 'Metadata' directory.

    Returns:
        str: The path of the metadata JSON file.
    """
    Metadatafile = Path(__file__).parent / "Metadata"
    for file in os.listdir(Metadatafile):
        if file.endswith(".json"):
            metadata_json_path = os.path.join(Metadatafile, file)
            print(metadata_json_path)
            return metadata_json_path

def GetTextFromJson(metadata_json_path):
    """
    Retrieves the description from a JSON file.

    Args:
        metadata_json_path (str): The path to the JSON file.

    Returns:
        str: The description extracted from the JSON file.
    """
    with open(metadata_json_path) as json_file:
        data = json.load(json_file)
        title = data["title"]
        description = data["description"]
        tags = data["tags"]
        return description
    
def GetCodeFromJson(metadata_json_path):
    """
    Extracts the code from a given metadata JSON file path.

    Args:
        metadata_json_path (str): The path to the metadata JSON file.

    Returns:
        str: The extracted code from the metadata JSON file.
    """
    metadata_json_path

    file_name = os.path.basename(metadata_json_path)

    cleaned_name = file_name.replace("MetaData_", "").replace(".json", "")

    print(cleaned_name)

    return cleaned_name

def dallE3(result_array, image_duration, topic):
    """
    Generate DALL-E images based on the given result array, image duration, and topic.

    Args:
        result_array (list): A list of items containing keywords and durations.
        image_duration (int): The duration of each image in seconds.
        topic (str): The topic to be included in the keywords.

    Returns:
        list: A list of tuples containing the generated image URLs and the corresponding keywords.

    """
    image_urls = []
    status = False
    for item in result_array:
        status = False
        row_urls = []
        print(item[1])
        while item[1] > 0:
            keywords_str = ' '.join(item[0])
            keywords_str = topic + " " + keywords_str
            print(keywords_str)
            images_required = min(int(item[1] / image_duration), 1)
            client = OpenAI()
            response = client.images.generate(
                model="dall-e-3",
                prompt=keywords_str,
                size="1024x1024",
                quality="standard",
                n=images_required,
            )
            row_urls.append(response.data[0].url)
            item[2] = response.data[0].url
            item[1] -= image_duration
            status = True
        if status:
            image_urls.append((row_urls, keywords_str))

    print(result_array)
    return image_urls

def SlideshowofImages(topic,key_points,video_length,lengths,image_duration):

    """
    Create a slideshow of images based on the given parameters.

    Args:
        topic (str): The topic of the slideshow.
        key_points (list): List of key points related to the topic.
        video_length (int): The desired length of the video in seconds.
        lengths (list): List of sentence lengths corresponding to the key points.
        image_duration (int): The duration in seconds for each image in the slideshow.

    Returns:
        tuple: A tuple containing the final video clip and the height of the video.

    """

 
    photos_path = Path(__file__).parent / "Photos" / topic

    photo_path_arr = []
    for file in os.listdir(photos_path):
        if file.lower().endswith((".jpeg", ".png", ".jpg")):
            photo_path = os.path.join(photos_path, file)
            photo_path_arr.append(photo_path)

    sentences_length=lengths
    sentences_length = [list(elem) for elem in sentences_length]
    for row in range (len(sentences_length)):
        if(sentences_length[row][1] > image_duration):
            length = sentences_length[row][1]
            mod = round((sentences_length[row][1] / image_duration))
            remainer = sentences_length[row][1] % image_duration    
            if(key_points[row] != 0):
                if(mod > 0):
                    if(remainer < image_duration/2):
                        sentences_length[row][1] = (mod)*image_duration
                    else:
                        sentences_length[row][1] = (mod+1)*image_duration
                    if(key_points[row] != 0):
                        sentences_length[row][0] = key_points[row]
                else:
                     sentences_length[row][1] = image_duration
                     sentences_length[row][0] = key_points[row]
            else:
                sentences_length[row][0]="default"
                    

    print("sentences length:" +str(sentences_length))
    result_array = []
    used_images = []
    for item in sentences_length:
        keywords, length = item[0], item[1]

        if isinstance(keywords, list):
            keywords = ' '.join(keywords)

        keyword_list = keywords.split()

        matched_images = []

        for keyword in keyword_list:
            match_count = 0

            for image_path in photo_path_arr:
                if keyword.lower() in image_path.lower() and image_path not in used_images and length > 0:
                    matched_images.append(image_path)
                    used_images.append(image_path)
                    length -= image_duration
                    match_count += 1

            if match_count == 0:
                base_keyword = keyword
                variation_suffix = 0

                while f"{base_keyword}_{variation_suffix}" in photo_path_arr and variation_suffix <= 5:
                    variation_suffix += 1

                if variation_suffix <= 5:
                    variation = f"{base_keyword}_{variation_suffix}"
                    matched_images.append(variation)
                    used_images.append(variation) 

            if length <= 0:
                break

        result_array.append([keyword_list, length, matched_images])

    for row in result_array:
        print(f"Keywords: {row[0]}, Remaining Length: {row[1]}, Matched Images: {row[2]}")

    photo_path_arr_final = []
    #call DallE3 to generate images based on remaining length and keywords
    if any(row[1] > 0 for row in result_array):
        urls = dallE3(result_array, image_duration, topic)
        new_filepaths = download_images(urls, photos_path)

        photo_path_arr_final = [image for row in result_array for image in row[2] if image.endswith((".jpeg", ".png", ".jpg"))]
        photo_path_arr_final.extend(new_filepaths)

        print(photo_path_arr_final)
    else:
        for row in result_array:
            for image in row[2]:
                if(image.endswith((".jpeg", ".png", ".jpg"))):
                    photo_path_arr_final.append(image)
        print(photo_path_arr_final)

    clips = [ImageClip(img, duration=image_duration) for img in photo_path_arr_final]

    video = concatenate_videoclips(clips, method="chain")
    video_clip = ImageSequenceClip(photo_path_arr_final, fps=25)
    #video resolution is 1920 height and 1080 width
    final_clip = clips_array([[video_clip], [video]], bg_color=(255,12,255),rows_widths=[ 1000], cols_widths=[ 1000])
    

    video_clip = video_clip.without_audio()
    video_height = video_clip.size[1]
    return final_clip, video_height

def download_images(image_urls, output_directory):
    """
    Downloads images from the given list of image URLs and saves them to the specified output directory.

    Args:
        image_urls (list): A list of tuples containing image URLs and corresponding keywords.
        output_directory (str): The directory where the downloaded images will be saved.

    Returns:
        list: A list of filepaths of the downloaded images.

    """
    new_filepaths = []
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for row_index, (row_urls, keywords_str) in enumerate(image_urls):
        for image_index, url in enumerate(row_urls):
            response = requests.get(url)
            if response.status_code == 200:
                file_extension = os.path.splitext(urlparse(url).path)[-1]
                filename = f"{keywords_str}_{image_index}{file_extension}"
                filepath = os.path.join(output_directory, filename)

                while os.path.exists(filepath):
                    keywords_str = ' '.join(keywords_str.split()[1:])
                    filename = f"{keywords_str}_{image_index}{file_extension}"
                    filepath = os.path.join(output_directory, filename)

                with open(filepath, "wb") as file:
                    file.write(response.content)
                    new_filepaths.append(filepath)
                print(f"Image saved: {filepath}")
            else:
                print(f"Failed to download image from {url}")
    return new_filepaths

def spacy_nlp(text, topic):
    """
    Perform natural language processing using Spacy on the given text.

    Args:
        text (str): The input text to be processed.
        topic (str): The topic related to the text.

    Returns:
        list: A list of arrays, where each array contains the important verbs found in a sentence.

    """
    nlp = spacy.load("en_core_web_sm")
    sentences = [sent.text for sent in nlp(text).sents]
    verbs_array = []
    for sentence in sentences:
        doc = nlp(sentence)
        important_verbs = [token.text for token in doc if token.pos_ == "VERB"]
        verbs_array.append(important_verbs)
    return verbs_array

def main():
    start_time = time.time()

    start_cpu_percent = psutil.cpu_percent()
    start_memory_info = psutil.virtual_memory()
    output_path_name = "Generated_videos"

    metadata_json_path = GetMetadata()
    text = GetTextFromJson(metadata_json_path)
    random_code = GetCodeFromJson(metadata_json_path)

    #use if you have api key
    #mp3_location, wav_location = SendAPIRequest(random_code,text)
    #use if you have existing audio for testing
    mp3_location, wav_location = UseExistingAudio()


    audio_clip, result = GetAudio(mp3_location, wav_location)
    indexesoffullstops = [i for i, x in enumerate(text) if x == "."]
    print(indexesoffullstops)
    subtitle_info, video_length,sentences_length = CalculateAudio(result,indexesoffullstops)

    #use for preselected video
    #video_clip, video_height = GetVideoClip(video_length)
    
    topic = "Sparta"
    key_points = spacy_nlp(text,topic)
    print(key_points)
    video_clip,video_height = SlideshowofImages(topic,key_points,video_length,sentences_length,image_duration=3)
    subtitle_clips = CreateSubtitles(subtitle_info, video_clip)

    GenerateVideo(video_clip, subtitle_clips, audio_clip, video_height,random_code,output_path_name)

    #measure_resource_usage(start_cpu_percent, start_memory_info,start_time)

if __name__ == '__main__':
    main()