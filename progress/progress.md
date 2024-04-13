## How to use COPILOT to explain stuff?
1. Select the class (function) you want to explain
2. Type /explain the class (function) in #file:your_file

## REPLACING RESNET WITH EFFICENT NET
Models use different feature sizes. In the downloaded config file for the model (the file that's named something like: "logs/2021-07-30T21-34-25_vggsound_transformer/configs/2021-07-30T21-34-25-project.yaml") I replaced the data.feat_depth from 2048 to 1792. This fixed an error during the post transofmration

Now getting this error when sampling codebook indices

![alt text](image.png)


The feature downloading feature is broken. When I run the script "download_vas_features.sh" it dumps everything to the same folder and overrides the already downloaded videos because they start from video_00000 for every class.

i want to run the tranformer training from scratch to see in what format does it need the features to be


removed the early stop callback
modified the feature extraction pipeline (added efficient net, added tiling of video to 10 seconds, mentioned in docs but was not in the code)

now monitoring training loss instead of validation loss
then switched to monitoring loss instead of training loss

added auto_scale_batch_size option (although it might be useless if batch size is set to 2 in the data config idk)


you can write about turso in coursework
also write about why you selected efficient net, some graphs, comparisons etc