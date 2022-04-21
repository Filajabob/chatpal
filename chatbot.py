import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import os
import pyjokes
import requests

# Stop debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()

with open("intents.json", 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({
            'intent': classes[r[0]],
            'probability': str(r[1])
        })

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I don't understand."

    if 'action' in i:
        return {
            "response": result,
            "action": i['action']
        }
    else:
        return {
            "response": result,
            "action": None
        }


def ask(message):
    ints = predict_class(message.lower())
    res = get_response(ints, intents)

    # for choice in ints:
    #     print(f"Intent: {choice['intent']}")
    #     print(f"Probability: {choice['probability']}")

    return res


print("Chatbot has loaded.")
print(f"To retrain the model, run 'python {os.path.abspath('training.py')}'")
print("To quit the session, hit Ctrl+C or just tell the chatbot like a human")

print("WARNING: The bot you are about to chat with cannot understand you. The bot uses pre-prepared patterns and "
      "responses, however the neural network classifies your response.\n")

while True:
    msg = input("You: ")
    resp = ask(msg)
    print("ChatPal: " + resp['response'])

    action = resp['action']

    if not action:
        # being extra efficient
        continue

    elif action == "joke":
        print('\t' + pyjokes.get_joke())

    elif action == "news":
        r = requests.get("https://newsapi.org/v2/top-headlines?country=ca&apiKey=16c500a83f6840c5abd51ea95ba88b2e").json()
        articles = r['articles']
        article = random.choice(articles)

        print(f"\n{article['title']} by {article['author']}\n")
        print(article['description'])
        print(f"Read more: {article['url']}\n")

    elif action == "quit":
        quit()
