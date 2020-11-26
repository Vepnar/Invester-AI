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