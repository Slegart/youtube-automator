import os
import subprocess

def call_combiner_whisper():

    combiner_whisper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'video_creator', 'combiner_whisper.py'))
    
    try:
        subprocess.run(['C:/Users/slega/AppData/Local/Microsoft/WindowsApps/python3.10.exe', combiner_whisper_path], check=True)

        print("combiner_whisper.py execution complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing combiner_whisper.py: {e}")

call_combiner_whisper()

print("continue")
