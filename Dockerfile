#Dockerfile for DSCI 522 Group Project Credit Card Default Prediction
#Author: Cici  Du, Tianwei Wang
#Created: 2021-12-09
#Last updated: #2021-12-09

ARG BASE_CONTAINER=continuumio/miniconda3
FROM $BASE_CONTAINER

RUN apt-get update -y

#install dev tools
RUN apt-get install gcc python3-dev chromium-driver -y

#install gnu make
RUN apt-get install make -y

RUN conda install --quiet -y -c conda-forge \
    "ipykernel=6.5.*" \
    "requests=2.24.*" \
    "vega_datasets=0.9.*" \
    "graphviz=2.49.*" \
    "jinja2=3.0.*" \
    "imbalanced-learn=0.8.*" \
    "altair=4.1.*" \
    "altair_saver=0.5.*" \
    "docopt=0.6.*" \
    "matplotlib=3.5.*" \
    "matplotlib-inline=0.1.*" \
    "numpy=1.21.*" \
    "pandas=1.3.*" \
    "pickleshare=0.7.*" \
    "scikit-learn=1.*" \
    "scipy=1.7.*" \
    "pyppeteer=0.2.*"

RUN pip install \
    "mglearn==0.1.9" \
    "psutil==5.8.0" \
    "selenium==3.141.0" \
    "dataframe-image==0.1.1" \
    "jupyter-book==0.12.1" \
    "xlrd==1.2.*" \
    "altair-data-server==0.4.*"

