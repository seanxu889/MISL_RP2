This repository holds the LISA_CNN attack system code (and some results generated by our own dataset) used in [Robust Physical-World Attacks on Deep Learning Visual Classification](https://arxiv.org/abs/1707.08945). The software carries an [MIT license](https://github.com/evtimovi/robust_physical_perturbations/blob/master/LICENSE).

<p align="center">
<img src="https://github.com/seanxu889/MISL_RP2/blob/master/PhysicalAttack%20note-1.jpg">
</p>

The folders are as follows:

* `lisa-cnn-attack` holds the code to attack the LISA-CNN that classifies US road signs from the LISA dataset. Contains a model that achieves 91% accuracy on that dataset. This is the most rudimentary implementation of the algorithm.

Further details are given in `README` files in the respective folders. They also specify how to download portions that are needed for the code to run but are not committed here due to size.

Note that in `lisa-cnn-attack` and in `gtsrb-cnn-attack` we include portions of an older version of the [cleverhans](https://github.com/tensorflow/cleverhans) library for compatibility. It carries its own [MIT License](https://github.com/tensorflow/cleverhans/blob/master/LICENSE).

## "subtle_poster" training preocess
Mask out the background region.
<p align="center">
<img src="https://github.com/seanxu889/MISL_RP2/blob/master/subtle_poster.png">
</p>

## "camouflage_art" training preocess
Only attack the weak regions based on "subtle_poster" result. Use the mask to project computed perturbations to a shape of sticker.
<p align="center">
<img src="https://github.com/seanxu889/MISL_RP2/blob/master/camouflage_art.png">
</p>
