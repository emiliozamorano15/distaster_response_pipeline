# Data Scientist Nanodegree:
# Disaster Response Pipeline Project

## Table of Contents

- [Overview](#overview)
- [Project Components](#components)
  - [ETL Pipeline](#etl)
  - [ML Pipeline](#ml)
  - [Flask Web App](#webapp)
- [Running](#run)
- [Conclusion](#conclusion)
- [File Structure](#files)
- [Software Requirements](#requirements)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

## 1. Overview
This project is one of the requirements for Udacity's Data Scientist Nanodegree Program. It consists on developing a ML model which is able to classify (as accurately as possible) text messages sent during disasters into response categories, and deploy it in a webapp. The dataset corresponds to real data gathered and classified by [Future Eight](https://www.figure-eight.com/) for this project.
The main objective is to provide assistance to response teams during disasters, by classifying texts from different sources in order to channel them to a corresponding agency/organization. In this way response times can be optimized and providing assistance can be expedited.

<a id='components'></a>
## 2. Components 

<a id='etl'></a>
### 2.1. ETL Pipeline

This pipeline corresponds to the data processing stage of the project. The code is contained in the _data/process_data.py_ file, which performs the following operations:
- Reads the `disaster_messages.csv` and `disaster_categories.csv` files and merges them into a single dataset
- Encodes the categories to binary category columns per message
- Stores the resulting dataframe into the `data/DisasterResponse.db` SQL database

<a id='ml'></a>
### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:
- Reads data from the database
- Splits data into training and test sets
- Cleans, tokenizes, and lemmatizes text messages
- Performs NLP transformations such as vectorizing token counts and calculating tf-idf
- Trains a Random Forest Classifier within a grid search for optimizing hyperparameters
-  Evaluates model performance
-  Stores a serialized version of the final model for the webapp

<a id='webapp'></a>
### 2.3. Web App
The webapp consists on a locally hosted Flask app containing two pages. The main page contains users controls to input messages for predicting:

**_Screenshot 1_**
![Landing page](https://github.com/emiliozamorano15/distaster_response_pipeline/blob/main/snapshots/main1.JPG)

Additionally the main page displays two visualizations from the training set:

**_Screenshot 2_**
![Visuals](https://github.com/emiliozamorano15/distaster_response_pipeline/blob/main/snapshots/main2.JPG)

Once given a message for prediction, the results are displayed on the second page. The message used as an input is displayed and the predicted categories are hightlighed in blue:

**_Screenshot 3_**
![Results](https://github.com/emiliozamorano15/distaster_response_pipeline/blob/main/snapshots/go1.JPG)

<a id='run'></a>
## 3. Running

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

<a id='conclusion'></a>
## 4. Conclusion

Due to the imbalance on the label categories certain considerations for measuring the performance of the prediction model. Measures such as accuracy show a higher value than other measures, *e.g.* recall. This needs to be taken into account when making decisions based on the predictions from this model. On the other hand, while training a model with imbalanced labels one could apply techniques such as over- or undersampling to adress this issue.

<a id='files'></a>
## 5. Files

<pre>
.
├── app
│   ├── run.py------------------------# Flask file that runs the webapp
│   └── templates
│       ├── go.html-------------------# Displays classification results
│       └── master.html---------------# Main page
├── data
│   ├── DisasterResponse.db-----------# Database to store processed data
│   ├── disaster_categories.csv-------# Raw Category Data
│   ├── disaster_messages.csv---------# Raz Message Data
│   └── process_data.py---------------# ETL Process
├── models
│   └── train_classifier.py-----------# Trains classification model
├── snapshots-------------------------# Snapshots of website
├── LICENSE---------------------------# License file
├── README.md-------------------------# Readme file
└── requirements.txt------------------# List of required packages
</pre>

<a id='requirements'></a>
## 6. Software Requirements
This project was developed with Python 3.7.10 for Windows. The required packages are listed under `requirements.txt`.

<a id='credits'></a>

## 7. Credits and Acknowledgements

Many thanks to [Future Eight](https://www.figure-eight.com/) and [Udacity](https://www.udacity.com) for setting this project requirements and providing the data.
Additional resources used to improve documentation and this README.md file can be found [here](https://medium.com/udacity/three-awesome-projects-from-udacitys-data-scientist-program-609ff0949bed).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
