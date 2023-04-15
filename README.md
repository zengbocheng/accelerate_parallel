# Introduction

This a demo of [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/v0.18.0/en/index) for pytorch parallel.

# Usage

1.install requirements
```shell script
pip install -r requirements
```

2.edit the config file in yamls/

3.create directories
```shell script
mkdir checkpoints figs logs
```

4.run the shell script to train and watch log file.
```shell script
nohup sh run.sh &
tail -f logs/out1.log
```