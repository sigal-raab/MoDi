# MoDi: Unconditional Motion Synthesis from Diverse Data

<!-- ![alt text](coming_soon.png) -->

![Python](https://img.shields.io/badge/Python->=3.6.10-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.5.0-Red?logo=pytorch)

This repository provides a library for unconditional motion synthesis from diverse data, as well as applications including interpolation and semantic editing in the latent space, inversion, and code for quantitative evaluation. It is based on our work [MoDi: Unconditional Motion Synthesis from Diverse Data](https://sigal-raab.github.io/MoDi.html).

<div>
  <span class="center">
    <img src="https://github.com/sigal-raab/sigal-raab.github.io/tree/main/images/jump_teaser.gif" style="width: 20vw;">
    <img src="images/walk_teaser.gif" style="width: 20vw;">
    <img src="images/dance_teaser.gif" style="width: 20vw;">
    <img src="images/salto_teaser.gif" style="width: 20vw;">
  </span>
</div>

The library is still under development.


## Prerequisites

This code has been tested under Ubuntu 16.04 and Cuda 10.2. Before starting, please configure your Conda environment by
~~~bash
conda env create --name MoDi --file environment.yaml
conda activate MoDi
~~~
or by 
~~~bash
conda create -n MoDi python=3.6.10
conda activate MoDi
conda install -y pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install -y -c conda-forge tqdm
conda install -y -c conda-forge opencv
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda pandas
conda install -y -c conda-forge scipy
conda install -y -c anaconda scikit-learn
pip install clearml
~~~


## Data 
### Pretrained Model 

We provide a pretrained model for motion synthesis. 


Download the [pretrained model](https://drive.google.com/file/d/1vGYam0B9LOb7IeWejgA-i2gtGY3loF4D/view?usp=sharing).
<!-- Download the [pretrained model](https://drive.google.com/file/d/1Ezd6fW33B4GjmycKPVO-s0yIBfnqu9T5/view?usp=sharing).  -->

Create a folder by the name `data` and place the downloaded model in it.


### Data for a Quick Start

We use the Mixamo dataset to train our model. You can download our preprocessed data from 
 [Google Drive](https://drive.google.com/file/d/1tloa9eXbnZj9IkdK01M96uw2GWgBhhQJ/view?usp=sharing) into the `data` folder. 
Unzip it using `gunzip`.

In order to know which Mixamo motions are held in the motions file, 
you may download the naming data from [Google Drive](https://drive.google.com/file/d/1g_NhADUjlEhNUUK0Utyycqj70u2Ewyzz/view?usp=sharing) as well. 
The naming data is a text file containing the character name, motion name, and path related info. 
In the given data, all motions are related to the character Jasper.

<!--In order to use motions for other characters in Mixamo, or for other datasets, please refer to our instructions in the [More Data]() section. <span style="color: red"> TBC </span>-->
 
The action recognition model used for evaluation is trained using the joint location data. You can download the Mixamo preprocessed joint location data from  
[Google Drive](https://drive.google.com/file/d/1hV8nxxfC5V-r9tZiTUthIO1g9lFik_Sh/view?usp=sharing).
You may also download the corresponding naming data from
[Google Drive](https://drive.google.com/file/d/1zXKuYPY-KAro--N9Wao0mMYgazJdpp2Q/view?usp=sharing).

## Novel motion synthesis

Here is an example for the creation of 18 random samples, to be placed in `<result path>`.

~~~bash
python generate.py --type sample --motions 18 --ckpt ./data/ckpt.pt --out_path <results path> --path ./data/edge_rot_data.npy
~~~

## Train from scratch

Following is a training example with the command line arguments that were used for training our best performing model. 

~~~bash
python train.py --path ./data/edge_rot_data.npy --skeleton --conv3fast --glob_pos --v2_contact_loss --normalize --use_velocity --foot --name <experiment name>
~~~

### Training on a new character.
After downloading a new character dataset, you can add it to the file `utils/config.yaml` with 
your new character name and the joints you wish to use, and add the flag `--character <your character>` to the train command.

## Interpolation in Latent Space 
Here is an example for the creation of 3 pairs of interpolated motions, with 5 motions in each interpolation sequence, to be placed in `<result path>`.
~~~bash
python generate.py --type interp --motions 5 --interp_seeds 12-3330,777,294-3 --ckpt ./data/ckpt.pt --out_path <results path> --path ./data/edge_rot_data.npy
~~~
The parameter interp_seeds is of the frame `<from_1[-to_1],from_2[-to_2],...>`.
It is a list of comma separated `from-to` numbers,
where each from/to is a number representing the seed of the random z that creates a motion. 
This seed is part of the file name of synthesised motions. See [Novel Motion Synthesis](##novel-motion-synthesis).
The `-to` is optional, and if it is not given, then our code interpolates the latent value of `from` to the average latent space value, aka truncation.
In the example above, the first given pair of seeds will induce an interpolation between the latent values related to the seeds 12 and 3330, 
and the second will induce an interpolation between the latent value related to the seeds 777 and to the mean latent value.

## Semantic Editing in Latent Space 
Following is an example for editing the `gradual right arm lifting` and `right arm elbow angle` attributes.
~~~bash
python latent_space_edit.py --model_path ./data/ckpt.pt --attr r_hand_lift_up r_elbow_angle --path ./data/edge_rot_data.npy 
~~~
Note that editing takes a long time, but once it is done, the data that was already produced can be reused, 
which significantly shortens the running time. See inline documentation for more details.

## Inversion
Following is and example for inversion of a motion using an optimizer: 
~~~bash
python inverse_optim.py --ckpt ./data/ckpt.pt --out_path <results path> --target_idx 32 --path ./data/edge_rot_data.npy
~~~

Use `--target_idx` to chose the index of a motion to invert from `edge_rot_data.npy` file. 

## Encoder
The encoder is able to invert motions into MoDi's latent space, enabling many applications. 

### Pretrained Encoder
Download the [pretrained encoder](https://drive.google.com/file/d/1-NsJ7RPcPc_e-DqaOUKM0_-RSzPIeHr2/view?usp=sharing).


### Applications

The encoder enables the following applications: Inversion, Motion Fusion, Spatial Editing, Denoising, Prediction from Prefix.

Download the encoder [test data](https://drive.google.com/file/d/1K6fPexKnWcue3_fYAbi_oSFq1nk4ihl_/view?usp=sharing) and place it under the `data` folder. Then, run the following:
~~~bash
 python generate_encoder.py --path ./data/test_edge_rot_data.npy --application inversion --ckpt_existing <pretrained_model_path> --ckpt <pretrained_encoder_path> --model_name <model name> --eval_id 34,54 --out_path <save_path>
~~~
Arguments:
* `--application` can be one of the following: `[inversion, fusion, editing, editing_seed, denoising, auto_regressive]`
* `--ckpt_existing` is given a path to the pretrained model.
* `--ckpt` is given a path to the pretrained encoder.
* `--eval_id` is a comma seperated list of indices from the test set that the encoder will be applied on. In the case of `editing_seed` application, `eval_id` is used as a seed for a generated motion. 


### Train the Encoder
To train the encoder from scratch download the [train](https://drive.google.com/file/d/13skti2Em_lQpAnM8pvTeR5ZfYF_G4_O2/view?usp=drive_link) and [test](https://drive.google.com/file/d/1K6fPexKnWcue3_fYAbi_oSFq1nk4ihl_/view?usp=sharing) data, place them under the `data` folder. Then, run the following command:
~~~bash
 python train_encoder.py --ckpt_existing=<pretrained_model_path> --name <experiment name> --path=./data/train_edge_rot_data.npy --n_latent_predict=2 --action_recog_model=evaluation/checkpoint_0300_globpos_acc_0.99.pth.tar --n_frames=0
~~~

Arguments:
* `--ckpt_existing` is given a path to the pretrained model.

## Quntitative evaluation 
The following metrics are computed during the evaluation: FID, KID, diversity, precision and recall.
you can use the `--fast` argument to skip the precision and recall calculation which may take a few minutes.

Here is an example of running evaluation for a model saved in `<model_ckpt>`, where `<dataset>` is the dataset the model was trained with and `<gt_data>` is the path to the data the action recognition model was trained with.
`--motions` is the number of motions that will be generated by the model for the evaluation.
~~~bash
python evaluate.py --ckpt <model_ckpt> --path <dataset> --act_rec_gt_path <gt_data>
~~~


## Visualisation

### Figures
We use figures for fast and easy visualization. Since they are static, they cannot reflect smoothness and naturalness of motion, hence we recommend using bvh visualization, detailed in the next paragraph.
The following figures are generated during the different runs and can be displayed with any suitable app:
- Motion synthesis and interpolation: file `generated.png` is generated in the folder given by the argument `--output_path`
- Training: files real_motion_`<iteration number>`.png and fake_motion_{}.png are generated in the folder `images` under the folder given by the argument `--output_path`.

### Using Blender to visualise bvh files
Basic acquaintance of Blender is expected from the reader. 

Edit the file `blender/import_multiple_bvh.py`, and set the values of the variables `base_path` and `cur_path`:
- Their concatenation should be a valid path in you file system.
- Any path containing bvh files would work. In particular, you would like to specify paths that were given as the 
`--output_path` arguments during motion synthesis, motion interpolation, inversion or latent space editing.
- `cur_path` will be displayed in blender.

There are two alternatives: run the script from commandline or interactively in Blender.
#### Running the script from commandline
~~~bash
blender -P blender/import_multiple_bvh.py
~~~

#### Interactively running the script in Blender
- Start the blender application.
- [Split](https://docs.blender.org/manual/en/latest/interface/window_system/areas.html?highlight=new%20window) one of the areas and turn it into a [text editor](https://docs.blender.org/manual/en/latest/editors/index.html).
- Upload blender/import_multiple_bvh.py. Make sure the variables `base_path` and `cur_path` are set accroding to your path, or set them now.
- Run blender/import_multiple_bvh.py.
- You can interactively drag the uploaded animations and upload animations from other paths now.


## Acknowledgements

Part of the code is adapted from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

Part of the code in `models` is adapted from [Ganimator](https://github.com/PeizhuoLi/ganimator).

Part of the code in `Motion` is adapted from [A Deep Learning Framework For Character Motion Synthesis and Editing](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing).

Part of the code in `evaluation` is adapted from [ACTOR](https://github.com/Mathux/ACTOR/blob/d3b0afe674e01fa2b65c89784816c3435df0a9a5/src/recognition/models/stgcn.py), [Action2Motion](https://github.com/EricGuo5513/action-to-motion), [Metrics for Evaluating GANs](https://github.com/abdulfatir/gan-metrics-pytorch), and [Assessing Generative Models via Precision and Recall](https://github.com/msmsajjadi/precision-recall-distributions).

Part of the training examples are taken from [Mixamo](http://mixamo.com).  

Part of the evaluation examples are taken from [HumanAct12](https://github.com/EricGuo5513/action-to-motion).

## Citation

If you use this code for your research, please cite our paper:

~~~bibtex
@inproceedings{raab2023modi,
  title={Modi: Unconditional motion synthesis from diverse data},
  author={Raab, Sigal and Leibovitch, Inbal and Li, Peizhuo and Aberman, Kfir and Sorkine-Hornung, Olga and Cohen-Or, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={13873--13883},
  year={2023}
}
