#!/usr/bin/env python3

import argparse
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer

PATH_TO_MODEL="../model/model.ckpt-200000"

synthesizer = Synthesizer()

if __name__ == '__main__':
  from wsgiref import simple_server
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
  print("================LOADED======================")
  synthesis = synthesizer.synthesize(" لَا تَثِقْ فِي كُلِّ مَا تَرَاهُ ")
  filename = "speech.wav"
  wav_file = open(filename, 'wb')
  wav_file.write(synthesis)
  wav_file.close()
  print("=========WROTE TO FILE============")
else:
  synthesizer.load(os.environ['CHECKPOINT'])

