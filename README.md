## This repo is a fork of VLMEvalKit to test our FineBench Dataset
Download the dataset from https://huggingface.co/datasets/FINEBENCH/FineBench

### Run FineBench

1. Install vlmeval kit following the file ***Official_README.md*** file
2. Put the FineBench Annotations in the directory where the AVA frames are extracted. Ex: `dataset_path/annotations/test_subset.json`
3. Update the file ***finebench_config.json*** with the path to the dataset and the model you want you run.
4. Run `bash run_finebench.sh`
