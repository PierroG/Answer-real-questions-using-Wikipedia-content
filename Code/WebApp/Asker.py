import torch
import transformers
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import spacy
import en_core_web_sm
import wikipedia
import wikipediaapi
from nltk.tokenize import word_tokenize
from langdetect import detect


class ask:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        #move model to device
        self.model = self.model.to(self.device)
        self.nlp = en_core_web_sm.load()
        self.wiki = wikipediaapi.Wikipedia(language='en')


   


    def generateAnswer(self, question, answer_text):
        # == Tokenize == Apply the tokenizer to the input text, treating them as a text-pair. (CPU)
        input_ids = self.tokenizer.encode(question, answer_text, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # == Set Segment IDs == Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1
        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a
        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input_ids)
        # == Run Model == Run our example through the model. (GPU)
        #move tensor to device
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        segment_ids_tensor = torch.tensor([segment_ids]).to(self.device)
        
        try:
            start_scores, end_scores = self.model(input_ids_tensor, # TheSSSS tokens representing our input text.
                                 token_type_ids=segment_ids_tensor) # The segment IDs to differentiate question from 
            
            # Find the tokens with the highest `start` and `end` scores.
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            # get score
            start_score = float(start_scores[0,answer_start])
            end_score = float(end_scores[0,answer_end])
        
        
            # == Print Answer without ## ==
            # Start with the first token.
            answer = tokens[answer_start]
    
            # Select the remaining answer tokens and join them with whitespace.
            for i in range(answer_start + 1, answer_end + 1):
        
                # If it's a subword token, then recombine it with the previous token.
                if tokens[i][0:2] == '##':
                    answer += tokens[i][2:]
        
                # Otherwise, add a space then the token.
                else:
                    answer += ' ' + tokens[i]

            return answer, start_score+end_score
    
        except RuntimeError:
            return "", 0

    def extract_subject_with_spacy(self, question, lang):
    
        #question = truecase.get_true_case(question) #le truecaser est un peu bidon j'ai l'impression
        #print(question)
        
        subject_dict = {'subject' : '', 'infos' : []} #dictionnaire qui contiendra le sujet, et les infos complémentaires
        if (lang == "fr"):
            osef_list = ['qui','pourquoi','quoi','quand','quel','quelle','comment','où','Qui','Pourquoi','Quoi','Quand','Quel','Quelle','Comment']
        else:
            osef_list = ['who','why','what','when','which','how', 'Who','Why','What','When','Which','How'] #noun to not take into account
        doc = self.nlp(question)
        
        #on prépare une liste des noms communs (ou plus précisent chunks, qui peuvent être des groupes nominaux plus larges, des unités de sens) de la question
        
        nouns_list = []
        #dep_list = []
        for noun in doc.noun_chunks :
            #dep_list.append(noun.root.dep_)
            nouns_list.append(noun)
               
        #on enlève de la liste des potentiels sujets les mots interrogatifs venant de osef_list
        for noun in nouns_list : 
            if str(noun) in osef_list : 
                nouns_list.remove(noun)
                    
        #on crée une liste d'entité nommées de la phrase. S'il y en a une dans la question, alors c'est le sujet
        #nn crée également une liste qui va contenir les labels de ces entités nommées, car certains types d'entités ne nous intéressent pas
        #les labels qui nous intéressent sont dans la liste relevant_labels
        
        #si il y a des entités nommées, on les utilise comme sujet et les chunks alentours comme infos supplémentaires
        #si il n'y a pas d'entités nommées, on va uniquement regarder les chunks (dans le 'else')
        
        ents_list = []
        labels_list = []
        
        relevant_labels = ['PERSON','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW']
        for ent in doc.ents :
            if ent.label_ in relevant_labels : 
                ents_list.append(ent.text)
                labels_list.append(ent.label_)
                #dep_list.append(ent.dep_) #pour tests sur depencies
                
        if ents_list and labels_list : 
            print(ents_list)
            #print(dep_list)
            print('subject found by ent : ', labels_list[-1] , ents_list[-1], '\n')
            subject_dict['subject'] = ents_list[-1] #on renvoie la dernière entité nommée pertinente trouvée
            for other_noun in nouns_list : 
                subject_dict['infos'].append(other_noun)
            #return(subject_dict)
            
    
        else : 
            
    #si notre liste de chunks potentiels sujets est vide : pas de sujet
    #si elle est égal à 1 : pas de doute, le sujet est cet élément
    #si elle est plus grande que 1, le sujet est le deuxième élément
    #règle simpliste mais qui semble suivre la logique de la formulation d'une question : c'est souvent le second nom qui est le sujet dans les questions qui en comportent deux, j'ai l'impression 
        
            print(nouns_list)
            #print(dep_list)
            if(len(nouns_list)) == 0 :
                print("subject not found, please try another formulation", '\n')

            else :
                print("subject found by noun: " + str(nouns_list[-1]), '\n')
                subject_dict['subject'] = str(nouns_list[-1]) #le sujet est le dernier chunk
                for other_noun in nouns_list[0:-1] : #dans ces cas de figure avec + d'un nom, il faudra quand même récupérer le nom qui n'est pas le sujet, pour aller l'utiliser en scrappant la page wiki du sujet
                    subject_dict['infos'].append(other_noun)
                #return(subject_dict)
        return(subject_dict)

    def get_sections_list(self, page):
        osef_list = ['Sources', 'Further reading', 'External links']
        def get_sections(sections, sections_list, level=0):
                for s in sections:
                        #print("%s: %s - %s" % ("*" * (level + 1), s.title, len(s.text)))
                        #check if there is text and if section is usefull
                        if len(s.text) != 0 and s.title not in osef_list:
                            sections_list.append(s.text)
                        get_sections(s.sections, sections_list, level + 1)
                        
        sections_list = []
        sections_list.append(page.summary)
        get_sections(page.sections, sections_list)
        return sections_list

    def get_paragraph(self, page):
        result = []
        result = self.get_sections_list(page)
        
        paragraph = []
        for section in result:
            for item in section.split("\n"):
                #check len <512 // 400-450
                if len(word_tokenize(item)) < 400:
                    paragraph.append(item)
        return paragraph

    def get_best_answer(self, question, subject):

        search = wikipedia.search(subject)
        if len(search) > 0 : 
            page = self.wiki.page(wikipedia.search(subject)[0])

            paragraph = self.get_paragraph(page)
        
        
            answers = []
            scores = []

            for p in paragraph[:15]:
                answer, score = self.generateAnswer(question, p)
                answers.append(answer)
                scores.append(score)
            
            max_value = max(scores)
        
            #index = scores.index(max_value)
            #return answers[index], paragraph[index], page.fullurl

            top = sorted(scores, reverse=True)[:3]
            index = [scores.index(i) for i in top]
            answer = [answers[i] for i in index]
            context = [paragraph[i] for i in index]
            return answer, context, page.fullurl
            
        else : 
               return("Your question doesn't seem to correspond to a specific subject. Could you try to reformulate ?", "No Wikipedia page found", "")

    def is_wh (self, question) : 
        doc = self.nlp(question)
        interrogatives = ['where', 'who','why','what','when','which','how', 'Where', 'Who','Why','What','When','Which', 'How']
        words = [word for word in doc]
        if words[0].pos_ == 'SPACE' : #si le premier token est un espace, on le supprime pour le traitement
            words = words[1:]
    
        if words[0].text in interrogatives or words[1].text in interrogatives : #si le premier ou deuxième mot de la question est un mot interrogatif, c'est une question en "wh-" : ex "When is Barack Obama born?", "In which country is Paris ?"
            return(True)
        else : 
            return(False)     

    def is_polar(self, question) : #questions yes/no
        doc = self.nlp(question)
        words = [word for word in doc]
        if words[0].pos_ == 'SPACE' : 
            words = words[1:]
    
        if words[0].pos_ == 'AUX' : #si la question commence par un auxiliaire, c'est une question yes/no
            return(True)
        else : 
            return(False)

    def is_pseudocleft (self, question) : #détecter les pseudocleft -> phrases qui ressemblent à des questions mais n'en sont pas : "Who is the President of Nicaragua doesn't interest me."    
        doc = self.nlp(question) 
        interrogatives = ['where', 'who','why','what','when','which','how', 'Where', 'Who','Why','What','When','Which', 'How']
        words = [word for word in doc]
        if words[0].pos_ == 'SPACE' : 
            words = words[1:]
    
        if words[0].text in interrogatives and words[0].head.dep_ in ["csubj", "advcl"] : #si la dépendence head du mot interrogatif est un adverbial clause ou un subject clause (en gros le sujet ou l'adverbe sont des clauses, genre le sujet de la question c'est le "What", comme dans la phrase "What she says is true"), alors on est face à un pseudocleft (ou pseudo-wh) -> c'est une règle linguistique        
            return(True) 
        else : 
            return(False)

    def is_tutorial (self, question) : #détecter les questions tutorielles
        doc = self.nlp(question)
        modals = ['to', 'can', 'could', 'shall', 'should']
        words = [word for word in doc]
        if words[0].text == "How" and words[1].text in modals  : 
            return(True)
        else : 
            return(False)
    
    def is_toapp (self, question) : #détecte les questions adressées à l'app
        doc = self.nlp(question)
        words = [word for word in doc]
        toapp_words = ['you', 'your', 'You', 'Your']
        is_toapp = False
        if doc.ents == () : 
            for word in words : 
                if word.text in toapp_words : 
                    is_toapp = True
        else : 
            for ent in doc.ents : 
                for word in words : 
                    if word.text in toapp_words : 
                        is_toapp = True
        return(is_toapp)

    def is_personal (self, question) : 
        doc= self.nlp(question)
        words = [word for word in doc]
        personal_words = ['I', 'i', 'My', 'my', 'Me', 'me', 'Myself', 'myself']
        is_personal = False
        in_quotes = False
        if doc.ents == () : 
            for word in words : 
                if word.text in personal_words and in_quotes == False : 
                    is_personal = True
                in_quotes = False
                if word.text == '"' : 
                    in_quotes = True
        else : 
            for ent in doc.ents : 
                for word in words : 
                    if word.text in personal_words and in_quotes == False : 
                        is_personal = True
                    in_quotes = False
                    if word.text == '"' :
                        in_quotes = False
        return(is_personal)
    
    def language_detection(self, question):
        if (question != ""):
            lang = detect(question)
            wikipedia.set_lang(lang)
            self.wiki = wikipediaapi.Wikipedia(language=lang)
        return lang

    def predifined_answer (self, question, lang) : #to compute the predifined answer given the question type
        
        if self.is_polar(question) :
            sujet = self.extract_subject_with_spacy(question, lang)
            a, p, u = self.get_best_answer(question, sujet['subject'])
            
            if not u : 
                u = "Sorry, we couldn't find any Wikipedia article related to your question."
            return(True, "You are expecting me to answer by yes or non. I can’t give an answer that is that precise. But here is information that will help you answer it yourself.",
                   "Maybe you should check that Wikipedia link.",
                   u )
        elif self.is_pseudocleft(question) :
            sujet = self.extract_subject_with_spacy(question)
            a, p, u = self.get_best_answer(question, sujet['subject'])
            if not u : 
                u = "Sorry, we couldn't find any Wikipedia article related to your question."
            return(True,"What you just typed looks like a question… but doesn’t seem to be one. Maybe you should try to rephrase it ?",
                   "But this Wikipedia page could still help you.",
                   u )
        elif self.is_tutorial(question) : 
            return(True, "It seems like you are asking for a tutorial. I can’t provide such information to you… maybe you should try to rephrase your question, or ask a more factual one.",
                   "If you really want a good tutorial, you can check on Wikihow",
                   "https://www.wikihow.com/Main-Page")
        elif self.is_personal(question) : 
            return(True, 'I can’t answer personal questions… Maybe you should try finding it by yourself. You can still ask me factual questions, and I’ll be glad to answer !', 
                  "Really, I can't.", 
                  "https://www.wikihow.com/Know-Your-True-Self") 
        elif self.is_toapp(question) : 
            return(True, "Are you trying to get information that concerns me ? I’m shy, so I won’t answer, sorry … Maybe you should try asking some general knowledge questions !",
                   "If you want to know more about me or my creators :",
                   "https://ask-me.azurewebsites.net/#about")
        else : 
            return(False, None, None, None)


    def run(self, question):
        lang = self.language_detection(question)
        isSpecial, specialAnswer, details, url = self.predifined_answer(question, lang)
        if isSpecial : 
            return(specialAnswer, details, url)    
        else : 
            sujet = self.extract_subject_with_spacy(question, lang)
            if sujet['subject'] == "": 
                return("Your question doesn't seem to correspond to a specific subject. Could you try to reformulate ?", "No Wikipedia page found","")
            else: 
                a, p, u = self.get_best_answer(question, sujet['subject'])
                return a, p, u


        
        

# if __name__=='__main__':
#     asker = ask()
#     a,s = asker.generateAnswer('who i am ?', 'i am me')
#     print(a)
#     question = "when was created the moon ?"
#     sujet = asker.extract_subject_with_spacy(question)
#     print(sujet['subject'])
#     a, p, u = asker.get_best_answer(question, sujet['subject'])
#     print(a)
#     print(u)
#     print(p)