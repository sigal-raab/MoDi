# MoDi: Unconditional Motion Synthesis from Diverse Data

<!-- ![alt text](coming_soon.png) -->

![Python](https://img.shields.io/badge/Python->=3.6.10-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.5.0-Red?logo=pytorch)

This repository provides a library for unconditional motion synthesis from diverse data, as well as applications including interpolation and semantic editing in latent space, inversion, and code for quantitative evaluation. It is based on our work [MoDi: Unconditional Motion Synthesis from Diverse Data](https://sigal-raab.github.io/MoDi.html).

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

Download the [pretrained model](https://drive.google.com/file/d/1edzdGtkjNSxnSl2ig2UeGnHRG_xcaSWG/view?usp=sharing). 

Create a folder by the name `data` and place the downloaded model in it.

Unzip the downloaded model using `gunzip`.

### Data for a Quick Start

We use Mixamo dataset to train our model. You can download our preprocessed data from 
 [Google Drive](https://drive.google.com/file/d/1tloa9eXbnZj9IkdK01M96uw2GWgBhhQJ/view?usp=sharing) into the `data` folder. 
Unzip it using `gunzip`.

In order to know which Mixamo motions are held in the motions file, 
you may download the naming data from [Goggle Drive](https://drive.google.com/file/d/1g_NhADUjlEhNUUK0Utyycqj70u2Ewyzz/view?usp=sharing) as well. 
The naming data is a text file containing the character name, motion name, and path related info. 
In the given data, all motions are related to the character Jasper.

In order to use motions for other characters in Mixamo, or for other datasets, please refer to our instructions in the [More Data]() section. <span style="color: red"> TBC </span>

<!-- [joint location data](https://drive.google.com/file/d/1hV8nxxfC5V-r9tZiTUthIO1g9lFik_Sh/view?usp=sharing)
[joint location names](https://drive.google.com/file/d/1zXKuYPY-KAro--N9Wao0mMYgazJdpp2Q/view?usp=sharing) -->

## Novel motion synthesis

Here is an example for the creation of 18 random samples, to be placed in <result path>.

~~~bash
python generate.py --type sample --motions 18 --ckpt ./data/ckpt.pt --out_path <results path> --path ./data/edge_rot_data.npy
~~~

## Train from scratch

Following is a training example with the command line arguments that were used for training our best performing model. 

~~~bash
python train.py --path ./data/edge_rot_data.npy --skeleton --joints_pool --glob_pos --v2_contact_loss --normalize --use_velocity --foot --name <experiment name>
~~~

## Interpolation in Latent Space 
Here is an example for the creation of 3 pairs of interpolated motions, with 5 motions in each interpolation sequence, to be placed in <result path>.
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
python latent_space_edit.py --model_path ./data/ckpt.pt --attr r_hand_lift_up r_elbow_angle
~~~
Note that editing takes a long time, but once it was done, that data that was already produced can be reused, 
which significantly shortens the running time. See inline documentation for more details.

## Inversion

~~~bash
python inverse_optim.py --ckpt ./data/ckpt.pt --out_path <results path> --target_idx 32
~~~

## Quntitative evaluation 

Under development.

## Visualisation

### Figures
We use figures for fast and easy visualization. Since they are static, they cannot reflect smoothness and naturalness of motion, hence we recommend using bvh visualization, detailed in the next paragraph.
The following figures are generated during the different runs and can be displayed with any suitable app:
- Motion synthesis and interpolation: file `generated.png` is generated in the folder given by the argument `--output_path`
- Training: files real_motion_<iteration number>.png and fake_motion_{}.png are generated in the folder `images` under the folder given by the argument `--output_path`.

### Using Blender to visualise bvh files
Basic acquantance of Blender is expected from the reader. 

Edit the file `blender/import_multiple_bvh.py`, and set the values of the variables `base_path` and `cur_path`:
- Their concatentaion should be a valid path in you file system.
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

Part of the code in 'evaluation' is adapted from [](), [](), and [](). <span style="color: red"> TBC </span>

Part of the training examples is taken from [Mixamo](http://mixamo.com).  

Part of the evaluation examples is taken from [](). <span style="color: red"> TBC </span>

## Citation

If you use this code for your research, please cite our paper:

~~~bibtex
@article{raab2022modi,
  title={MoDi: Unconditional Motion Synthesis from Diverse Data},
  author={Raab, Sigal and Leibovitch, Inbal and Li, Peizhuo and Aberman, Kfir and Sorkine-Hornung, Olga and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2206.08010},
  year={2022}
}
