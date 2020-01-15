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

def work_loop():
  while(True):
    if not (os.listdir(TEXT_DIR)):
      print("Text directory empty!")
      time.sleep(5)
    else:
      print("Text directory is not empty!")
      time.sleep(5)

def save_wav_file(audio_data):
  filename = "speech" + int(time.time()) + ".wav"
  wav_file_path = os.path.join(SPEECH_DIR, filename)
  wav_file = open(wav_file_path, "wb")
  wav_file.write(audio_data)
  wav_file.close()
  print(f"Wrote to {wav_file_path}")

def synthesize_from_file(file):
  pass

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
  synthesis = synthesizer.synthesize(" لَا تَثِقْ فِي كُلِّ مَا تَرَاهُ ")
  save_wav_file(synthesis)
  work_loop()
else:
  synthesizer.load(os.environ['CHECKPOINT'])

