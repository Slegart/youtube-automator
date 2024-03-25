Youtube Content Automator creating a short video based on user inputs.
See VideoCreator/Generated_videos to see a basic example.

Usage:
1) Adjust paths.
2) Enter desired video options through videoCreator/Tkinter/ContentAddTkinter.py. It will create a metadata file that will be later used to generate video.
3) Install dependencies and run combiner_whisper.py under video_creator.
4) Modules are working separately but for extended automation add the following scripts to main/mainrunner.py such as youtube_uploader_selenium/uploader.py to use selenium follow account binding steps here https://github.com/linouk23/youtube_uploader_selenium

Important points:
1) You must have an OpenAI API key to generate text-to-speech & image files. Other free options can also be used for text-to-speech.
2) To use your images instead of ai generated ones you must open a folder that matches the video tag under Images and use that folder to store images.
3) In default config OpenAi text-to-speech does not provide subtitles, so subtitles are generated separately using whisper.ai. Slight errors might happen although it's rare.
4) Main dependencies such as gecko driver, ImageMagick, etc. are included in the package but in the case of an error, you might need to manually download them and add path variables.
  

Workflow:
1) Video title/text content/tags are entered through GUI in videoCreator/Tkinter/ContentAddTkinter.py.
2) Running combiner_whisper.py will generate mp3 file for video & wav type of same audio for subtitles.
3) Content is split into sections depending on overall length, Keywords in each section are spotted by spacy, and timeframes are assigned so subtitles match the generated images.
4) Video is saved under the Generated_videos folder.
