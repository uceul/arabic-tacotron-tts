#!/usr/bin/env python3

import argparse
from hparams import hparams, hparams_debug_string
import os
import time
from synthesizer import Synthesizer

PATH_TO_MODEL = "../model/model.ckpt-200000"
TEXT_DIR = "test/text"
SPEECH_DIR = "test/speech"

synthesizer = Synthesizer()

def work_loop(synthesizer):
  while(True):
    files = os.listdir(TEXT_DIR)
    if not files:
      print("Text directory empty!")
      time.sleep(5)
    else:
      print("Text directory is not empty!")
      text_file_name = get_oldest_file(files)
      text_file = open(text_file_name, "r")
      text = ""
      if text_file.mode == "r":
        text = text_file.read()
      else:
        raise IOError("Failed to read file %s" % text_file_name)
      speech = synthesizer.synthesize(text)
      save_wav_file(speech)
      os.remove(text_file_name)

def save_wav_file(audio_data):
  timestamp = int(time.time())
  filename = "speech" + str(timestamp) + ".wav"
  wav_file_path = os.path.join(SPEECH_DIR, filename)
  wav_file = open(wav_file_path, "wb")
  wav_file.write(audio_data)
  wav_file.close()
  print("Wrote to %s" % wav_file_path)

def synthesize_from_file(file):
  pass

def get_oldest_file(files):
  now = time.time()
  relative_path = os.path.join(TEXT_DIR, files[0])
  abs_path = os.path.abspath(relative_path)
  oldest = files[0], now - os.path.getctime(abs_path)

  for file in files[1:]:
    relative_path = os.path.join(TEXT_DIR, file)
    abs_path = os.path.abspath(relative_path)
    age = now - os.path.getctime(abs_path)
    if age > oldest[1]:
      oldest = file, age
  return os.path.join(TEXT_DIR, oldest[0])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=PATH_TO_MODEL)
  parser.add_argument('--port', type=int, default=9200)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  print(hparams_debug_string())
  synthesizer.load(args.checkpoint)
  print("================MODEL LOADED======================")
  work_loop(synthesizer)
else:
  synthesizer.load(os.environ['CHECKPOINT'])

