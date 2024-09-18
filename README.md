# Aspect Based-SA-on-Airline-Passenger-Reviews

This study aims to conduct Aspect-Based Sentiment Analysis (ABSA) on airline passengers' reviews to detect and assess the sentiment polarity associated with various service aspects, such as food and beverages quality, staff sympathy, inflight entertainment, and seat comfort. We scraped over 110,000 reviews from the Skytrax website, focusing on major airlines worldwide. Our methodology involves a two-step approach: Aspect Extraction and Aspect Sentiment 
Classification. Initially, we tested several models, including LDA (Latent Dirichlet Allocation), BERTopic, and Lemma Keywords Detection for aspect extraction, ultimately selecting the Lemma Keywords Detection method as the basis for sentiment analysis. In the sentiment classification phase, both supervised (e.g. BERT-based models) and unsupervised models (e.g. VADER) were evaluated and compared against a baseline. While the baseline model performed well across the dataset, supervised models, particularly BERT-based ones, showed significant improvements, especially on reviews with mixed sentiment across aspects. Our final models were applied to assess customer satisfaction across five key service aspects for major US airlines, demonstrating the potential of our approach to provide actionable insights into airline service quality. Despite some limitations in the aspect extraction process, the study highlights the effectiveness of the developed pipeline in performing ABSA on airline reviews.

<img src="https://media.giphy.com/media/NqhohLDKCaixsl2Ygb/giphy.gif" width="200">
<img src="https://media.giphy.com/media/3o6nV8OYdUhiuKja1i/giphy.gif?cid=ecf05e47dj7xh7hw8btbg0oc9ivfmup9xsgwhd7jy37xw2dk&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="200">

## Materials:
We worked on python 3.9 in Anaconda Jupyter Notebook. The following libraries were used in addition to the classical python libraries: Librosa, os, tensorflow, keras, scipy, soundfile, tqdm (to monitor progression of the code completion).

The up-to-date skytrax database is available following the link below: 
https://www.airlinequality.com/review-pages/a-z-airline-reviews/

Hardware requirement:
To train the deep learning models on PyTorch, the following GPU was used: NVIDIA GeForce RTX 3050, associated with cuda version 11.8 

![image](https://github.com/JadeArpaliangeas/Speech-Emotion-Recognition-CNN/assets/149436763/f37a04d3-81ba-4503-ad73-5e6a080b55e6)

## Conclusion: 
