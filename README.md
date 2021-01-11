# Ask.me : Question Answering using BERT and Wikipedia

Student exploration of Question Answering.

Our project aims to answer general knowledge questions by using a BERT model pretrained on SQuAD.

The pipeline can be summed-up in 3 essential steps : 
- Subject extraction
- Wikipedia article retrieval
- Answer extraction 

Our code is functional and we deployed a web app that can be accessed here : https://ask-me.azurewebsites.net/


The app can be launched locally with Code/WebApp/run_app.py with the following requirements : 
flask==1.1.2
torch==1.6.0
transformers==3.3.1
scikit-learn==0.22.1
nltk==3.4.5
spacy==2.3.2
wikipedia==1.4.0
Wikipedia-API==0.5.4
langdetect==1.0.8
en_core_web_sm==2.3.1

A docker image for the app is available in Results/AzureAskme




Pierre GONCALVES, Frederic ASSMUS, Axel DIDIER 

M2 NLP - Université de Lorraine - 2020
