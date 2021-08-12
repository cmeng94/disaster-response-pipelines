# Disaster Response Pipelines

## Table of Contents
1. [Project Description](#intro)
2. [File Descriptions](#files)
3. [Getting Started](#start)
4. [Usage](#use)
5. [Contact](#contact)
6. [Acknowledgement and License](#acknowledge)


<a id='intro'></a>
## 1. Project Description
The goal of this project is to analyze disaster message data and build a model that classifies disaster messages. The initial data sets contain pre-labeled messages that were sent during disaster events. This project uses NLP (natural language processing) and ML (machine learning) techniques to process and categorize disaster response messages.

Components of this project include
* an ETL pipeline that prepares and cleans data, 
* an ML pipeline that transforms data using NLP techniques and builds an ML classification model, and 
* a Web App that displays data visualizations and classifies new disaster messages.    
<img width="800" alt="screenshot1" src="https://user-images.githubusercontent.com/11303419/129129421-ec44aa2e-e08a-4033-85b6-4a6b55c37556.png">


<a id='files'></a>
## 2. File Descriptions
* Data files: `data/disaster_messages.csv` and `data/disaster_categories.csv`
* The ETL pipeline for data cleaning is prepared in `ETL_pipeline.ipynb` and then modularized in `data/process_data.py` for terminal execution.
* The ML pipeline for data transfomation and classification is prepared in `ML_pipeline.ipynb` and then modularized in `models/train_classifier.py` and `models/custom_transformer.py` for terminal execution.
* The Web App is contained in `app`, where `app/run.py` is the main execution file, and `app/templates` contains html files.

<a id='start'></a>
## 3. Getting Started
### Dependencies
The code is developed with Python 3.9 and is dependent on a number of python packages listed in `requirements.txt`.

### Installation
To run the code locally, clone this GitHub repository using the following code in terminal:
```sh
git clone https://github.com/cmeng94/disaster-response-pipelines
```

<a id='use'></a>
## 4. Usage

<a id='contact'></a>
## 5. Contact

<a id='acknowledge'></a>
## 6. Acknowledgement and License
