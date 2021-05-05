# Data Scientist Nanodegree: Disaster Response Pipeline Project

## Table of Contents

- [Project Overview](#overview)
- [Project Components](#components)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Running](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [File Structure](#files)
- [Software Requirements](#requirements)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

## 1. Project Overview


[Here](#eg) are a few screenshots of the web app.

<a id='components'></a>

## 2. Project Components


<a id='etl_pipeline'></a>
### 2.1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

<a id='ml_pipeline'></a>
### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

<a id='flask'></a>

### 2.3. Web App

<a id='eg'></a>

Running [this command](#com) **from app directory** will start the web app where users can enter their query, i.e., a request message sent during a natural disaster, e.g. _"Please, we need tents and water. We are in Silo, Thank you!"_.

**_Screenshot 1_**

![Landing page](https://github.com/emiliozamorano15/distaster_response_pipeline/blob/main/snapshots/main1.JPG)

What the app will do is that it will classify the text message into categories so that appropriate relief agency can be reached out for help.

**_Screenshot 2_**

![Visuals](https://github.com/emiliozamorano15/distaster_response_pipeline/blob/main/snapshots/main2.JPG)

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

## 4. Conclusion

As we can see the data is highly imbalanced. Though the accuracy metric is [high](#acc) (you will see the exact value after the model is trained by grid search, it is ~0.94), it has a poor value for recall (~0.6). So, take appropriate measures when using this model for decision-making process at a larger scale or in a production environment.

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
└── snapshots-------------------------# Snapshots of website

</pre>

<a id='requirements'></a>

## 6. Software Requirements


<a id='credits'></a>

## 7. Credits and Acknowledgements

Thanks <a href="https://www.udacity.com" target="_blank">Udacity</a> for letting me use their logo as favicon for this web app.

Another <a href="https://medium.com/udacity/three-awesome-projects-from-udacitys-data-scientist-program-609ff0949bed" target="_blank">blog post</a> was a great motivation to improve my documentation. This post discusses some of the cool projects from <a href="https://in.udacity.com/course/data-scientist-nanodegree--nd025" target="_blank">Data Scientist Nanodegree</a> students. This really shows how far we can go if we apply the concepts learned beyond the classroom content to build something that inspire others.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
