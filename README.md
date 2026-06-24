# Geometric Graph Learning for Credit Risk Detection

The software and data in this repository are the materials used in the research paper.

## Computing Requirements
To run the code inside this repository, we highly recommend to use `Conda`, an open-source package and environment management system for `Python`. For installing `Conda`, please refer to the website https://www.anaconda.com/products/distribution. Our code runs under Python 3.13, with the following dependencies need to be installed:

* numpy 2.4.1
* pytorch 2.8.0
* scipy 1.17.0
* torch-geometric 2.7.0

Note: Installing torch_geometric package requires matched CUDA and pytorch versions to run, please install the following matching packages first (torch-cluster, torch-scatter, torch-sparse, torch-spline-conv): https://pytorch-geometric.com/whl/index.html. Our code is run under the GPU of GeForce RTX 4080, with CUDA Version 12.8. For package implementation details, see torch_geometric's documentation: https://pytorch-geometric.readthedocs.io/en/latest/index.html.

## File Structures

We provide a simple demo, the primary file for implementation are listed as follows:

* **model_geo.py**: Framework of the proposed framework.
* **demo.py**: Model training.
* **utils.py**: Reusable modules for model training and evaluation.
* **GeoGData.py**: Data loading and preparation.

## Citations

Citation format to be made.