import flask
app = flask.Flask(__name__)

@app.route('/')
def home():
    return flask.render_template('home.html', name=home)

#@app.route('/name')
#def name():
#    return 'second page'


if __name__ == '__main__':
    app.debug=True
    app.run()