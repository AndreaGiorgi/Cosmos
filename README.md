# Cosmos: A Machine Learning System for Exoplanet Classification

This master thesis project is focused on implementing a three model solution for exoplanet classification using TESS data. This project introduces a complex concurrent ETL Architectures which enables to:
* Load data from unstructured folders
* Select .fits files
* Load each .fits and extract time and light flux data
* Apply Phase Folding algorithm on data and generate a Global View of candidate exoplanet transit
* Put refined data into TFRecord.

There are three deep neural networks in this system:
* Cosmos-SNN: a Self Normalized Network dedicated to classify categorical data
* Cosmos-SCNN: a Self Normalized Convolutional Network dedicated to classify Global Views
* Cosmos-Hybrid: a two arms Neural Network composed by concatenating Cosmos-SNN and Cosmos-SCNN. 

Requirements
------------

This project requires some specific modules, defined inside the requirements.txt file

How to use
------------

Using an IDE, like vscode or PyCharm, open super_coordinator.py and run.
In order to generate only KFold results open neural_network_coordinator.py and run.

Future features
------------

* One shot classification providing data id and predicted label
* Anomaly detection network (Bayesian Network)

Maintainers
------------

- Andrea Giorgi

