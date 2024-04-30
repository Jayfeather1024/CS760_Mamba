# CS760_Mamba

## Environment Installation
Step by step guide to prepare the environment for running Mamba.

```
git clone https://github.com/Jayfeather1024/CS760_Mamba
cd CS760_Mamba
conda create -n mamba python=3.10.13
conda activate mamba
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install packaging
pip install causal-conv1d
pip install mamba-ssm
pip install transformers
pip install -e 3rdparty/lm-evaluation-harness
pip install git+https://github.com/bigscience-workshop/promptsource.git
```

## Scripts for Evaluation
Model Performance Evaluation (results saved in `model_performance_evaluation.log`)
```
bash model_performance_evaluation.sh > model_performance_evaluation.log
```

Inference Time Evaluation (results saved in `generation_speed_evaluation.log`)
```
python generation_speed_evaluation.py > generation_speed_evaluation.log
```