# Aspect-Based Sentiment Analysis for Analyzing Airline Passenger Reviews

<img src="https://media.giphy.com/media/p8HvKv9rCWQ3k3Shzb/giphy.gif?cid=ecf05e474ffyw041n6ie0eccy21fys86hxb4wqdmibl03cq8&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="200">

## Project Overview

### Objectives
This study aims to conduct Aspect-Based Sentiment Analysis (ABSA) on airline passenger reviews to detect and assess the sentiment polarity associated with various service aspects, such as:
- Food and beverages quality
- Staff sympathy
- Inflight entertainment
- Seat comfort

We scraped over **110,000 reviews** from the Skytrax website, focusing on major airlines worldwide.

Our methodology involves a two-step approach:
1. **Aspect Extraction**
2. **Aspect Sentiment Classification**

Initially, we tested several models, including LDA, BERTopic, and Lemma Keywords Detection for aspect extraction, ultimately selecting the Lemma Keywords Detection method as the basis for sentiment analysis. In the sentiment classification phase, both supervised (BERT-based models) and unsupervised models (e.g., VADER) were evaluated and compared to a baseline pre-trained model for ABSA.

### Results
While the baseline model performed well across the dataset, supervised models showed significant improvements, especially on reviews with mixed sentiment (bipolar data). Our final models were applied to assess customer satisfaction across five key service aspects for major US airlines, demonstrating the potential of our approach to provide actionable insights into airline service quality. 

Despite some limitations in the aspect extraction process, the study highlights the effectiveness of the developed pipeline in performing ABSA on airline reviews.

<img src="https://media.giphy.com/media/NqhohLDKCaixsl2Ygb/giphy.gif" width="200">

## Repository Contents

- **Report Paper**:  
  [Aspect-Based Sentiment Analysis Methods for Analyzing Airline Passenger Reviews Report.pdf](https://github.com/JadeArpaliangeas/Aspect-Based-SA-on-Airline-Passenger-Reviews/blob/dbb48cb7d18f5409c49707d8cf3306bcfd0fb7e6/Aspect-Based%20Sentiment%20Analysis%20Methods%20for%20Analyzing%20Airline%20Passenger%20Reviews%20Report.pdf)  
  Describes all the study and results.

- **PowerPoint Slides**:  
  [Presentation Aspect-Based Sentiment Analysis Methods for Analyzing Airline Passenger Reviews.pptx](https://github.com/JadeArpaliangeas/Aspect-Based-SA-on-Airline-Passenger-Reviews/blob/dbb48cb7d18f5409c49707d8cf3306bcfd0fb7e6/Presentation%20ABSA%20for%20Airline%20Passenger%20reviews.pptx)  
  Slides for oral presentation.

- **Notebooks Folder**:  
  This folder contains all the Jupyter Notebooks used for the project, exploring different stages of data processing, model training, and evaluation.
  - **File**: [Notebooks_descriptions.xlsx]([link-to-your-notebooks-descriptions](https://github.com/JadeArpaliangeas/Aspect-Based-SA-on-Airline-Passenger-Reviews/blob/dbb48cb7d18f5409c49707d8cf3306bcfd0fb7e6/Notebooks_descriptions.xlsx
))  
    Contains concise descriptions of the notebooks and their purpose, along with related functions to import from `commented_code_M2_thesis.py`.

- **Source Code (src)**:  
  - **File**: `commented_code_M2_thesis.py`  
    Compiles the most useful functions across the Notebooks.
  
  - **File**: `text_processing1.py`  
    Contains the `lemmatize_sentences` function (using parallel computing).
  
- **Requirements**:  
  - **File**: [requirements.txt](https://github.com/JadeArpaliangeas/Aspect-Based-SA-on-Airline-Passenger-Reviews/blob/dbb48cb7d18f5409c49707d8cf3306bcfd0fb7e6/requirements.txt)  
    Lists all necessary Python packages required to run the project.

### Requirements
- Python 3.9, Jupyter Notebook
- To train the deep learning models on PyTorch, the following GPU was used: **NVIDIA GeForce RTX 3050**, associated with **CUDA version 11.8**.
- See `requirements.txt` for more library requirements.

### Data Source
The up-to-date Skytrax database is available at the following link:  
[Skytrax Airline Reviews](https://www.airlinequality.com/review-pages/a-z-airline-reviews/)

### Contact
For any questions or suggestions, please reach out to:  
- Jade Arpaliangeas  
- **Email**: jade.arpaliangeas@gmail.com


