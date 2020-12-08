import flask
from flask import request, redirect
import Asker


app = flask.Flask(__name__)
asker = Asker.ask()

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        req = request.form
        question = req.get("search")
        if len(question) != 0:
            answer, context, url = asker.run(question)
            isStr = isinstance(answer, str)
            #answer = text
            #for i in range(1,50):
            #    answer, s = asker.generateAnswer(text,"Donald John Trump (born June 14, 1946) is the 45th and current president of the United States. Before entering politics, he was a businessman and television personality. Born and raised in Queens, New York City, Trump attended Fordham University for two years and received a bachelor's degree in economics from the Wharton School of the University of Pennsylvania. He became president of his father's real estate business in 1971, renamed it The Trump Organization, and expanded its operations to building or renovating skyscrapers, hotels, casinos, and golf courses. Trump later started various side ventures, mostly by licensing his name. Trump and his businesses have been involved in more than 4,000 state and federal legal actions, including six bankruptcies. He owned the Miss Universe brand of beauty pageants from 1996 to 2015, and produced and hosted the reality television series The Apprentice from 2004 to 2015.")
            return flask.render_template("home.html", isStr=isStr, answer=answer, question=question, context=context, url=url)
    return flask.render_template('home.html', name=home)

#@app.route('/name')
#def name():
#    return 'second page'


if __name__ == '__main__':
    app.run(debug=True)
