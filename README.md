# Invester-AI

The goal of this project is to create a machine learning algorihm that is able to trade cryptocurrency.

## Cloning the project

`git clone git@github.com:Vepnar/Invester-AI.git`

`cd Invester-AI`

## Configuring enviroment variables:

Create a file called `.env` and enter the following. 
Replace `[REDACTED]` key with the api key you obtained from [financialmodelingprep](https://financialmodelingprep.com/).

```
DATASET_DIR=./dataset
TRAIN_ON_SETS=BTC,XMR
KEY=[REDACTED]
COMPARING_CURRENCY=USD
```

## Downloading & converting dataset

Build the download container and run it.
Building the container the first time will take a while..

`docker-compose up --build download`

Run the following script to convert the downloaded files into usefull CSV files.

`docker-compose up --build convert`
