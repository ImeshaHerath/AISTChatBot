# main.py
from flask import Flask, render_template, request, jsonify

import chat

app = Flask(__name__)


@app.route("/")
def index_get():
    return render_template("base.html")

@app.route("/books")
def books():
    return render_template("books.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = chat.get_response(text)
    message = {"answer": response}  # dictionary
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
