# Import Libraries
import numpy as np
# import random
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from nltk_util import bag_of_words, tokenize, stem
from model import NeuralNetworks

# Open our intents.json file with read mode
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Create empty array for all words
all_words = []

# Create empty list for tags
tags = []

# Create empty list for hold both our patterns and the tags
xy = []


# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    # loop over all the patterns
    for pattern in intent['patterns']:
        # tokenize each word in the sentence(patterns)
        w = tokenize(pattern)
        # add tokenized patterns to our all_words list
        all_words.extend(w)  # use 'extend' coz we don't want any arrays in our all_word array
        # add to xy pair
        xy.append((w, tag))  # pattern and the corresponding tag


# Ignore punctuations
ignore_words = ['?', '.', '!', ',']

# stem and lower each word
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []  # bag of words
y_train = []  # tags
# loop over xy - pattern and the corresponding tag
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence(tokenized)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)  # each tag getting index as 0,1,2..
    y_train.append(label)


# Converted to the numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):  # access dataset with index
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])  # bag of words - len of all words
hidden_size = 16
output_size = len(tags)
# print(input_size, len(all_words))
# print(output_size, len(tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNetworks(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        # predicted output and the actual labels

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # backpropagation
        optimizer.step()

    # print every 100 epoch
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')  # loss at the end

# safe the data in dictionary
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}


FILE = "data_model.pth"
torch.save(data, FILE)
print(f'training complete files saved to {FILE}')