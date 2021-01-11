# Ask.me : Question Answering using BERT and Wikipedia

Student exploration of Question Answering.

Our project aims to answer general knowledge questions by using Wikipedia data and the following BERT model pre-trained on SQuAD :<br/>
bert-large-uncased-whole-word-masking-finetuned-squad.

this model is available here : https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

## The pipeline

The pipeline can be summed-up in 3 essential steps : 

<ul>
    <li>Subject extraction</li>
    <li>Wikipedia article retrieval</li>
    <li>Answer extraction </li>
</ul>

## Run

### Local

To run this project locally, you have download the WebApp file located in /Code/WebApp/<br/>
and run the file run_app.py with Python.

### Online 

Our code is functional and we deployed a web app that can be accessed here : https://ask-me.azurewebsites.net/

## Requirements

The app can be launched locally with the following requirements : <br/>
flask==1.1.2<br/>
torch==1.6.0<br/>
transformers==3.3.1<br/>
scikit-learn==0.22.1<br/>
nltk==3.4.5<br/>
spacy==2.3.2<br/>
wikipedia==1.4.0<br/>
Wikipedia-API==0.5.4<br/>
langdetect==1.0.8<br/>
en_core_web_sm==2.3.1<br/>

## Docker ready

The AzureAskme file with its Dockerfile can be used to create an image of our application. It will be ready to be deployed in a web service.

AzureAskme is available in /Results/

## Created by

Pierre GONCALVES, Frederic ASSMUS, Axel DIDIER 

M2 NLP - Université de Lorraine - 2020
