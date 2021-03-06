# Disaster Response Pipelines

## Table of Contents
1. [Project Description](#intro)
2. [File Descriptions](#files)
3. [Getting Started](#start)
4. [Web App Usage](#use)
5. [Contact](#contact)
6. [Acknowledgement and Licensing](#acknowledge)


<a id='intro'></a>
## 1. Project Description
The goal of this project is to analyze disaster message data and build a model that classifies disaster messages. The initial data sets contain pre-labeled messages that were sent during disaster events. This project uses NLP (natural language processing) and ML (machine learning) techniques to process and categorize disaster response messages.

Components of this project include
* an ETL pipeline that cleans data and loads it into a database, 
* an ML pipeline that transforms data using NLP and builds an ML classification model, and 
* a Web App that displays data visualizations and classifies new disaster messages.    
<img width="800" alt="screenshot1" src="https://user-images.githubusercontent.com/11303419/129129421-ec44aa2e-e08a-4033-85b6-4a6b55c37556.png">


<a id='files'></a>
## 2. File Descriptions
* Data files: `data/disaster_messages.csv` and `data/disaster_categories.csv`
* The ETL pipeline for data cleaning and database loading is prepared in `ETL_pipeline.ipynb` and then modularized in `data/process_data.py` for terminal execution.
* The ML pipeline for transforming data and building classification model is prepared in `ML_pipeline.ipynb` and then modularized in `models/train_classifier.py` and `models/custom_transformer.py` for terminal execution.
* The Web App is contained in `app`, where `app/run.py` is the main execution file, and `app/templates` contains html files.

<a id='start'></a>
## 3. Getting Started
### Dependencies
The code is developed with Python 3.9 and is dependent on a number of python packages listed in `requirements.txt`. To install required packages, run the following line in terminal:
```sh
pip3 install -r requirements.txt
```

### Installation
To run the code locally, create a copy of this GitHub repository by running the following code in terminal:
```sh
git clone https://github.com/cmeng94/disaster-response-pipelines
```
### Execution
* To run ETL pipeline that cleans data and loads database:
```sh
python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
* To run ML pipeline that trains classifier and saves model:
```sh
python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
* To run the Web App, first execute the following line, then visit [http://0.0.0.0:3001/](http://0.0.0.0:3001/).
```sh
python3 app/run.py
```

<a id='use'></a>
## 4. Web App Usage
With the Web App running, the user can submit a disaster message using the text box and click the `"Classify Message"` button to get classifcation results. For example, after submitting the message `"We are trapped in house with no food or water!"`:
<img width="800" alt="screenshot2" src="https://user-images.githubusercontent.com/11303419/129133187-837253f2-e19b-4147-b8e3-251546af436f.png">

the model classifies it to be in categories `"Related"`, `"Request"`, `"Aid Related"`, `"Water"` and `"Food"`:
<img width="800" alt="screenshot3" src="https://user-images.githubusercontent.com/11303419/129133493-c3144e46-8948-4f4e-b1ef-4825fdd55e47.png">

<a id='contact'></a>
## 5. Contact
**Chang Meng**
* Email: chang_meng@live.com
* Website: [https://sites.google.com/view/changmeng](https://sites.google.com/view/changmeng)

<a id='acknowledge'></a>
## 6. Acknowledgement and Licensing
This project is part of the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program at [Udacity](https://www.udacity.com/). Disaster data is provided by [Figure Eight](https://appen.com/).

