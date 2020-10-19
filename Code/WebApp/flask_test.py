import flask
from flask import request, redirect


app = flask.Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":

        req = request.form
        text = req.get("search")

        if text:
            answer = text
            return flask.render_template("home.html", answer=answer)

    return flask.render_template('home.html', name=home)

#@app.route('/name')
#def name():
#    return 'second page'


if __name__ == '__main__':
    app.debug=True
    app.run()