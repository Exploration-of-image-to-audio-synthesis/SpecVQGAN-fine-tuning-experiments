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
video_path = './data/vggsound/video/-Qowmc0P9ic_34000_44000.mp4'

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
    plt.show()

# Prepare Input
batch = default_collate([visual_features])
batch['feature'] = batch['feature'].to(device)
c = sampler.get_input(sampler.cond_stage_key, batch)


if check_video_for_audio(video_path):
    # Extract Spectrogram
    audio_fps = 22050
    spectrogram = extract_melspectrogram(video_path, audio_fps)
    spectrogram = {'input': spectrogram}
    # [80, 860] -> [80, 848]
    random_crop = False
    crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
    spectrogram = crop_img_fn(spectrogram)

    # Prepare input
    batch = default_collate([spectrogram])
    batch['image'] = batch['input'].to(device)
    x = sampler.get_input(sampler.first_stage_key, batch)

    # Encode and Decode the Spectrogram
    with torch.no_grad():
        quant_z, z_indices = sampler.encode_to_z(x)
        xrec = sampler.first_stage_model.decode(quant_z)

    print('Original Spectrogram:')
    display.display(tensor_to_plt(x, flip_dims=(2,)))
    print('Reconstructed Spectrogram:')
    display.display(tensor_to_plt(xrec, flip_dims=(2,)))
    plt.close()
    plt.close()
else:
    print('Could not find an audio track in the video file...')



# Define Sampling Parameters
W_scale = 1
mode = 'full'
temperature = 1.0
top_x = sampler.first_stage_model.quantize.n_e // 2
update_every = 0  # use > 0 value, e.g. 15, to see the progress of generation (slows down the sampling speed)
full_att_mat = True

# Start sampling
with torch.no_grad():
    start_t = time.time()

    quant_c, c_indices = sampler.encode_to_c(c)
    # crec = sampler.cond_stage_model.decode(quant_c)

    patch_size_i = 5
    patch_size_j = 53

    B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, 53*W_scale)

    z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

    if mode == 'full':
        start_step = 0
    else:
        start_step = (patch_size_j // 2) * patch_size_i
        z_pred_indices[:, :start_step] = z_indices[:, :start_step]

    pbar = tqdm(range(start_step, hr_w * hr_h), desc='Sampling Codebook Indices')
    for step in pbar:
        i = step % hr_h
        j = step // hr_h

        i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
        j_start = min(max(0, j - (patch_size_j // 2)), hr_w - patch_size_j)
        i_end = i_start + patch_size_i
        j_end = j_start + patch_size_j

        local_i = i - i_start
        local_j = j - j_start

        patch_2d_shape = (B, D, patch_size_i, patch_size_j)

        pbar.set_postfix(
            Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})'
        )

        patch = z_pred_indices \
            .reshape(B, hr_w, hr_h) \
            .permute(0, 2, 1)[:, i_start:i_end, j_start:j_end].permute(0, 2, 1) \
            .reshape(B, patch_size_i * patch_size_j)

        # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
        cpatch = c_indices
        logits, _, attention = sampler.transformer(patch[:, :-1], cpatch)
        # remove conditioning
        logits = logits[:, -patch_size_j*patch_size_i:, :]

        local_pos_in_flat = local_j * patch_size_i + local_i
        logits = logits[:, local_pos_in_flat, :]

        logits = logits / temperature
        logits = sampler.top_k_logits(logits, top_x)

        # apply softmax to convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1)
        z_pred_indices[:, j * hr_h + i] = ix

        if update_every > 0 and step % update_every == 0:
            z_pred_img = sampler.decode_to_img(z_pred_indices, sampling_shape)
            # fliping the spectrogram just for illustration purposes (low freqs to bottom, high - top)
            z_pred_img_st = tensor_to_plt(z_pred_img, flip_dims=(2,))
            display.clear_output(wait=True)
            display.display(z_pred_img_st)

            if full_att_mat:
                att_plot = all_attention_to_st(attention, placeholders=None, scale_by_prior=True)
                display.display(att_plot)
                plt.close()
            else:
                quant_z_shape = sampling_shape
                c_length = cpatch.shape[-1]
                quant_c_shape = quant_c.shape
                c_att_plot, z_att_plot = last_attention_to_st(
                    attention, local_pos_in_flat, c_length, sampler.first_stage_permuter,
                    sampler.cond_stage_permuter, quant_c_shape, patch_2d_shape,
                    placeholders=None, flip_c_dims=None, flip_z_dims=(2,))
                display.display(c_att_plot)
                display.display(z_att_plot)
                plt.close()
                plt.close()
            plt.close()

    # quant_z_shape = sampling_shape
    z_pred_img = sampler.decode_to_img(z_pred_indices, sampling_shape)

    # showing the final image
    z_pred_img_st = tensor_to_plt(z_pred_img, flip_dims=(2,))
    display.clear_output(wait=True)
    display.display(z_pred_img_st)

    if full_att_mat:
        att_plot = all_attention_to_st(attention, placeholders=None, scale_by_prior=True)
        display.display(att_plot)
        plt.close()
    else:
        quant_z_shape = sampling_shape
        c_length = cpatch.shape[-1]
        quant_c_shape = quant_c.shape
        c_att_plot, z_att_plot = last_attention_to_st(
            attention, local_pos_in_flat, c_length, sampler.first_stage_permuter,
            sampler.cond_stage_permuter, quant_c_shape, patch_2d_shape,
            placeholders=None, flip_c_dims=None, flip_z_dims=(2,)
        )
        display.display(c_att_plot)
        display.display(z_att_plot)
        plt.close()
        plt.close()
    plt.close()

    print(f'Sampling Time: {time.time() - start_t:3.2f} seconds')
    waves = spec_to_audio_to_st(z_pred_img, config.data.params.spec_dir_path,
                                config.data.params.sample_rate, show_griffin_lim=False,
                                vocoder=melgan, show_in_st=False)
    print(f'Sampling Time (with vocoder): {time.time() - start_t:3.2f} seconds')
    print(f'Generated: {len(waves["vocoder"]) / config.data.params.sample_rate:.2f} seconds')

    # Melception opinion on the class distribution of the generated sample
    topk_preds = get_class_preditions(z_pred_img, melception)
    print(topk_preds)

save_path = os.path.join(log_dir, Path(video_path).stem + '.wav')
soundfile.write(save_path, waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
print(f'The sample has been saved @ {save_path}')
display_audio.Audio(save_path)