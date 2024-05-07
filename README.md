# Align MacridVAE dataset generation code

This repository contains the code to generate the datasets used in the [AlignMacrid VAE](https://github.com/igui/Align-MacridVAE) model presented at [ECIR 2024](https://www.ecir2024.org/) in this [paper](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_5). The resulting dataset is published in Kaggle in https://www.kaggle.com/datasets/ignacioavas/alignmacrid-vae. 

## Install requirements

This code was tested in Python 3.10. We depend on some libraries like CLIP and so on. Before starting, install the required libraries using `pip install -r requirements.txt`

## How to generate datasets

First execute one of the following notebooks to download raw reviews, item information and images. Depending on the dataset you want to generate, you would have to execute one of the following notebooks:
- `1a_download_amazon_data.ipynb` to download reviews and images for the Amazon Datasets
- `1b_download_bookcrossing_data.ipynb` to download the Bookcrossing dataset
- `1c_download_ml_data.ipynb` to download any of the Movielens datasets

To keep the notebooks smaller, the code for processing datasets is contained in three modules in the same folder `amazon_dataset.py`, `bookcrossing_dataset.py` and `movielens_dataset.py`. Each of them exposes functions to load the items, images and reviews as pandas dataframes.
