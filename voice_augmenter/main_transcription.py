import os
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import numpy as np


def change_pitch(sound, factor):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


def change_speed(audio, speed_factor):
    new_frame_rate = int(audio.frame_rate * speed_factor)
    return audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})


def change_volume(audio, volume_change_db):
    return audio + volume_change_db


def modify_audio(input_path, output_path, speed_factor, pitch_shift_factor, volume_change_db, bit_rate="47k"):
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(input_path)

    # Change speed using pydub
    audio = change_speed(audio, speed_factor)
    audio = audio.set_frame_rate(16000)  # Ensure consistent frame rate

    # Change volume
    audio = change_volume(audio, volume_change_db)

    # Save intermediate WAV for Praat processing
    temp_wav_path = "temp.wav"
    audio.export(temp_wav_path, format="wav")

    # Load WAV with parselmouth
    sound = parselmouth.Sound(temp_wav_path)

    # Change pitch using Praat
    sound = change_pitch(sound, pitch_shift_factor)

    # Save the modified audio as WAV
    sound.save(temp_wav_path, "WAV")

    # Convert back to MP3 with specified bit rate
    modified_audio = AudioSegment.from_wav(temp_wav_path)
    modified_audio.export(output_path, format="mp3", bitrate=bit_rate)
    os.remove(temp_wav_path)


def augment_dataset(input_folder, output_folder, num_variations=10, bit_rate="47k"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            for i in range(num_variations):
                # Randomly select speed factor, pitch shift factor, and volume change
                speed_factor = np.random.uniform(1.0, 1.25)
                pitch_shift_factor = np.random.uniform(0.5, 1.5)
                volume_change_db = np.random.uniform(-20.0, 5.0)  # Volume change in decibels

                print(i)

                # Generate output path
                output_path = os.path.join(output_folder, f"{base_name}_var{i + 1}.mp3")

                # Modify audio
                modify_audio(input_path, output_path, speed_factor, pitch_shift_factor, volume_change_db, bit_rate)


# Example usage
input_folder = "H:\\PycharmProjects\\sound_labeler\\audioclips_mini"
output_folder = "output_transcription"
augment_dataset(input_folder, output_folder)
