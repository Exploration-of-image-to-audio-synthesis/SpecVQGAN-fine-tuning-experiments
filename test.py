import os
import time
from pathlib import Path

import IPython.display as display_audio
import soundfile
import torch
from IPython import display
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid
from tqdm import tqdm

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from feature_extraction.demo_utils import (ExtractResNet50, check_video_for_audio,
                                           extract_melspectrogram, load_model,
                                           show_grid, trim_video)
from sample_visualization import (all_attention_to_st, get_class_preditions,
                                  last_attention_to_st, spec_to_audio_to_st,
                                  tensor_to_plt)
from specvqgan.data.vggsound import CropImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_name = '2021-07-30T21-34-25_vggsound_transformer'
log_dir = './logs'
config, sampler, melgan, melception = load_model(model_name, log_dir, device)


# Select a video
video_path = '.\\data\\vggsound\\video\\-Qowmc0P9ic_34000_44000.mp4'

# Trim the video
start_sec = 0  # to start with 01:35 use 95 seconds
video_path = trim_video(video_path, start_sec, trim_duration=10)

# Extract Features
extraction_fps = 21.5
feature_extractor = ExtractResNet50(extraction_fps, config.data.params, device)
visual_features, resampled_frames = feature_extractor(video_path)

# Show the selected frames to extract features for
if not config.data.params.replace_feats_with_random:
    fig = show_grid(make_grid(resampled_frames))
    fig.show()

# Prepare Input
batch = default_collate([visual_features])
batch['feature'] = batch['feature'].to(device)
c = sampler.get_input(sampler.cond_stage_key, batch)