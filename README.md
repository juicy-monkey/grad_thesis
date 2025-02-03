# Graduate Thesis - Experiments

This repository includes the code for the Chapter 5 experiment of the graduate thesis found in **\[🌐🌐🌐.🌐🌐🌐🌐🌐🌐🌐🌐🌐.🌐🌐\]**.

The purpose of this experiment is to study how the integration of Transformer models with U-Net impact segmentation performance, specifically examining whether a non-pretrained Transformer can enhance segmentation performance on a small dataset.

The code is written in Python, and main libraries used include PyTorch, Torchvision, seg-metrics, and Matplotlib. PyTorch is used for developing the models by using predefined and custom components. Torchvision, an extension of PyTorch for computer vision applications, provides pre-coded models (such as the Vision Transformer) and functions for image transformations. The seg-metrics library evaluates image segmentation models, with Dice and the 95th percentile Hausdorff Distance used for validation in this experiment. Matplotlib is used to visualize model performance with different input images.

## Files

`data` includes only the sanity dataset used by `sanity_tests.ipynb`. Other dataset objects and images would be stored in this directory; however, the files are too large to be included in this repository.

`models` includes the U-Net, ViT-UNet, and UNETR models, whose performance is compared. The trained models are stored in this directory; however, they are too large to be included in this repository.

`dataset.py` defines the segmentation dataset class and functions for reading image files into objects. These objects are stored using [pickle](https://docs.python.org/3/library/pickle.html).

`model_training.ipynb` is a Jupyter notebook used to train the models.

`results.ipynb` is a Jupyter notebook that calculates the results of the trained models using the [seg_metrics](https://github.com/Jingnan-Jia/segmentation_metrics) package.

`sanity_tests.ipynb` is a Jupyter notebook used to validate that the models can overfit a single image. This sanity test helps quickly evaluate whether a model could be suitable.

`train.py` defines the training loop for training the models.

`utils.py` contains utility functions.

## How to run

If you want to train the models, download the images from [this link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data) and save each patient's directory to the `data/BrainMRISegmentation/kaggle_3m/` folder. Then, run the `create_dataset_files` function in the `dataset.py` file to generate objects from the images.

After that, you can train the models using the `model_training.ipynb` notebook and compute the results with the `results.ipynb` notebook.
