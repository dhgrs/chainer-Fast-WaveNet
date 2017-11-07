# chainer-Fast-WaveNet
A Chainer implementation of Fast WaveNet( https://arxiv.org/abs/1611.09482 ).

# Requirements
- Python3
- chainer v3
- librosa

# Usage
## set parameters
Edit `opt.py` before training.

## training

```
(without GPUs)
python train.py

(with GPU #n)
python train.py -g n

(with GPU #n and #m)
python train.py -g n -G m
```

## generating
```
python generate.py <trained model> <how long to generate(sec)>
```

If you get errors, please check that `opt.py` is same as you trained the model. Because `generate.py` uses setting in `opt.py` .
