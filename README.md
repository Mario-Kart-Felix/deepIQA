# deepIQA

This is the reference implementation of [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment][arxiv].
The pretrained models contained in the models directory were trained for both NR and FR IQA and for both model variants described in the paper.
They were trained on the full LIVE or TID2013 database respectively, as used in the cross-dataset evaluations. This evaluation script uses non-overlapping 32x32 patches to produce deterministic scores, whereas the evaluation in the paper uses randomly sampled overlapping patches.

> usage: evaluate.py [-h] [--model MODEL] [--top {patchwise,weighted}]
>                   [--gpu GPU]
>                   INPUT [REF]

## Dependencies
* [chainer](http://chainer.org/)
* scikit-learn
* opencv

## TODO
* add training code
* add cpu support (minor change)
* remove opencv and scikit-learn dependencies for loading data (minor changes)

[arxiv]: http://arxiv.org/abs/1612.01697

## Fork Version Modifications
- change some codes of [nr_model.py](./nr_model.py) to match the newest chainer (4.1.0) api
- add [evaluate_list_nr.py](./evaluate_list_nr.py) for evaluating a list of images
    - usage example

        ```console
        python evaluate_list_nr.py --img_dir dir_of_imgs --img_list path_to_img_list --model models/nr_live_weighted.model --gpu 0
        ```