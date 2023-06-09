# Introduction

This a demo of [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/v0.18.0/en/index) for pytorch parallel running in **single machine** with multiple GPUs. See [project](https://github.com/zengbocheng/accelerate_parallel/tree/multi_nodes) for the demo running in **multiple machines** with multiple GPUs. The only difference between two demos is yamls/accelerate.yaml.

# Usage

1.install requirements
```shell script
pip install -r requirements.txt
```

2.edit the config file of yamls/train.yaml

3.generate the config file of Accelerate
```shell script
sh config.sh
```
Note: if you run multiple Accelerate projects in the same machine, you may get the error that the address have been used already. You can try add 'main_process_port: 12388' in the accelerate.yaml to configure a different port.

4.create directories
```shell script
mkdir checkpoints figs logs
```

5.run the shell script to train and watch log file.
```shell script
nohup sh train.sh & disown
tail -f logs/out1.log
```