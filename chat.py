# Import Libraries
import random
import json
import torch

from model import NeuralNetworks
from nltk_util import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open our intents in read mode
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetworks(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
# evaluate mode
model.eval()

bot_name = "RJ"

def get_response(msg):
    # tokenized
    sentence = tokenize(msg)
    # Create bag of words
    X = bag_of_words(sentence, all_words)
    # reshape by giving 1 row and num of columns
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    # give predictions
    _, predicted = torch.max(output, dim=1)

    # getting actual tag
    tag = tags[predicted.item()]

    # use softmax to get actual probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # find the corresponding intent
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
