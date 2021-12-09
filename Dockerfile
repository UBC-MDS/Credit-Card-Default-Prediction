#Dockerfile for DSCI 522 Group Project Credit Card Default Prediction
#Author: Cici  Du
#Created: 2021-12-09
#Last updated: #2021-12-09

ARG BASE_CONTAINER=continuumio/miniconda3
FROM $BASE_CONTAINER

RUN apt-get update -y

#install dev tools
RUN apt-get install gcc python3-dev -y

#install gnu make
RUN apt-get install make -y

RUN conda install --quiet -y -c conda-forge \
    "ipykernel=6.6.*" \
    "requests=2.26.*" \
    "graphviz=2.50.*" \
    "jinja2=2.11.*" \
    "imbalanced-learn=0.8.*" \
    "altair=4.1.*" \
    "docopt=0.6.*" \
    "matplotlib=3.5.*" \
    "matplotlib-inline=0.1.*" \
    "numpy=1.20.*" \
    "pandas=1.3.*" \
    "pickleshare=0.7.*" \
    "scikit-learn=1.0.*" \
    "scipy=1.7.*"

RUN pip install \
	"mglearn==0.1.*" \
	"psutil==5.8.*" \
	"selenium==4.1.*" \
	"dataframe-image==0.1.*" \
    "vega-datasets==0.9.*" \
    "altair_saver==0.5.*" 
