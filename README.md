# SIMS
Semi-parametric Image Synthesis
# Photographic Image Synthesis with Cascaded Refinement Networks

This is a Tensorflow implementation of cascaded refinement networks to synthesize photographic images from semantic layouts.

<img src="./overallpipeline.pdf"/>

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.2) + Scipy + Numpy + Pillow + OpenCV.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.

### Quick Start (Setup)
1. Clone this repository.
2. cd into the cloned folder
3. Download the pretrained models from "https:/trainedmodels". It takes several minutes to download all the models.
4. Download the test data from "https:/testdata". It requires approximate 11G memory.
5. Download the training data from "https:/traindata". It requires approximate 60G memory

### Testing pipelines
Use resolution "512 x 1024" as a running example.
1. cd "matlab_code".
2. run "test_generate_transform.m" to generate testing data for the transformation network. The generated data is stored in  "../testdata/transform/"
3. cd "../python_code"
4. run "transformation.py" to generate the transformed results. The results are saved in folder "../result/transform"
5. cd "../matlab_code"
6.  run "test_generate_order.m" to generate testing data for the ordering network. The generated data is stored in "../testdata/order/".
7. cd "../python_code"
8. run "order.py", the order prediction is stored in folder "../result/order/data/".
9. cd "../matlab_code"
10. run "test_generate_canvas.m" to generate the canvas for the synthesis network to work on. The generated data is stored in "../testdata/synthesis".
11. cd "../python_code"
12. run "synthesis_512_1024.py" to generate the final results. The result is saved in folder "../result/synthesis".

One should notice one can skip steps "3", "4" and "5" and modify the "test_generate_canvas.m" according if you do not want to use spatial transformer. We do not observe significant improvement with spatial transformation on cityscapes dataset but it is reuired on "NYU" datasets since "NYU" has large variation of viewpoints.

### Training
1. Transformation network
(1) cd "matlab_code"
(2) run "test_generate_transform.m" to generate training data for the transformation network, training data is stored in folder "../traindata/transform/"
(3) cd "../python-code"
(4) run "transformation.py" with setting "training_phase = True"
2. Ordering network
(1) cd "matlab_code"
(2) run "train_generate_order.m" to generate training data for the ordering network, training data is stored in folder "../traindata/order/".
(3) cd "../python_code"
(4) run "ordering.py" with setting "training_phase = True". The model is saved in "../trainedmodels/order/"
3. Syntheiss network
The synthesis network is trained in a progessive way. We first train a model with resolution "256 x 512", and use it to initialize the model for resolution "512 x 1024", and then "1024 x 2048". The script for different resolutions is in file "synthesis_256_512.py", "synthesis_512_1024.py" and "synthesis_1024_2048.py" respectively.
(1) cd "../matlab_code/"
(2) run "train_generate_synthesis.m", training data is saved in folder "../traindata/synthesis/".
(3) run "synthesis_512_1024.py" with setting "training_phase = True". The result is saved in "../trainedmodels/synthesis/".

## Video
https://youtu.be/0fhUJT21-bs

## All Results
Results for all the datasets is stored in folder "../all_results".

## Citation
If you use our code for research, please cite our paper:

Xiaojuan Qi, Qifeng Chen, Jiaya Jia Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.


## Todo List
1.  Replace "transformation.py" with "appearance flow" [Zhou et al. 2016].
2. Add "contextual loss" [Mechrez et al. 2018] in the synthesis network to further improve the results.

## Question
If you have any question or request about the code and data, please email me at qxj0125@gmail.com . If you need more information for other datasets plesase send email. 

## License
MIT License