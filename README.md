## This repo is a fork of VLMEvalKit to test our FineBench Dataset


### Run FineBench

1. Install vlmeval kit following the file ***Official_README.md*** file
2. Download the AVA dataset and extract frames at 1FPS: Example for 1 frame: `dataset_path/frames/{video_id}/0001.jpg`
3. Put the FineBench Annotations in the directory where the AVA frames are extracted. Ex: `dataset_path/annotations/test_subset.json`
4. Update the file ***finebench_config.json*** with the path to the dataset and the model you want you run.
5. Run `bash run_finebench.sh`
