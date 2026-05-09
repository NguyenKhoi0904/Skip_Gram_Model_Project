import re
import string

import nltk
from nltk import sent_tokenize, word_tokenize

import numpy as np
from numpy.linalg import norm

import pandas as pd
from matplotlib import pyplot as plt

import pyvi
from pyvi import ViTokenizer

FILE_PATH = "vietnamese_dataset_on_wiki.txt"
STOP_WORD_PATH = "vietnamese-stopwords-dash.txt"
WINDOW_SIZE = 2
LEARNING_RATE = 0.01
EPOCH = 1
EMBEDDING_SIZE = 300

def load_file(url:str = FILE_PATH) -> str:
    with open(url, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def load_stop_word_data(url:str|None = STOP_WORD_PATH) -> list:
    with open(url, "r", encoding= "utf-8") as file:
        text = file.read()
    return text

def text_preprocessing(text:str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def separate_sentences_and_words(text:str|list, stop_word: list) -> list:
    sentences = sent_tokenize(text)
    sentences_preprocessed = [text_preprocessing(sentence) for sentence in sentences]
    words = [word_tokenize(ViTokenizer.tokenize(sentence))  for sentence in sentences_preprocessed]
    for i, sentence in enumerate(words):
            words[i] = [word.lower() for word in sentence if word not in stop_word]

    return words

def make_dictionary(words: list) -> dict:
    m = dict()
    index = 0
    for word in words:
        for w in word:
            if w in m :
                continue
            m[w] = index
            index += 1
    return m

def make_training_data(words: list, m:dict , window_size:int = WINDOW_SIZE)->list:
    td = list()
    for word in words:
        length = len(word)
        # print(word)
        for index in range(length):
            for j in range(max(index-window_size, 0), min(window_size+index+1, length)):
                if j == index :
                    continue
                td.append((m[word[index]], m[word[j]]))
    
    return td

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

def initialization_embedding_matrix(m: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(m)
    
    w1 = np.random.rand(n, EMBEDDING_SIZE)
    w2 = np.random.rand(EMBEDDING_SIZE, n)
    return w1, w2
    ...

def train(w1, w2, m: dict, training_data: list) -> np.array:
    total_words = len(m)
    print(f"W1: {w1.shape}")
    print(f"W2: {w2.shape}")
    for epoch in range(1,EPOCH+1):
        loss = 0
        for center_word, context_word in training_data:
            
            y_center_word = np.zeros(total_words)
            y_center_word[center_word] = 1
            
            y_context_word = np.zeros(total_words)
            y_context_word[context_word] = 1
            
            h = np.dot(y_center_word , w1)
            u = np.dot(h, w2)
            y_pred = softmax(u)
            
            loss -= (np.log(y_pred[context_word]))
            
            error = y_pred - y_context_word

            dL_dh = np.dot(w2, error)
            dL_dw2 = np.outer(h, error)
            w1[center_word] = w1[center_word] - LEARNING_RATE * dL_dh
            w2 = w2 - LEARNING_RATE * dL_dw2
        print(f"Epoch: {epoch}, loss: {loss:.4f}")
        with open("loss.txt", "a") as file:
            file.write(f"{epoch} {loss:.4f}\n")
            
    return w1, w2
    ...

def cosine_similarity(arr1, arr2):
    return np.dot(arr1, arr2)/(norm(arr1)*norm(arr2))
    ...

def word_similarity(w1 , m: dict):
    a = input("Nhập từ thứ nhất: ").strip().lower()
    b = input("Nhập từ thứ hai: ").strip().lower()
    
    a = ViTokenizer.tokenize(a)
    b = ViTokenizer.tokenize(b)
    if a not in m:
        print(f"{a} không có từ điển")
        return
    if b not in m:
        print(f"{b} không có từ điển")
        return
    
    a_vector_embedding = w1[m[a]]
    b_vector_embedding = w1[m[b]]
    
    cosine = cosine_similarity(a_vector_embedding, b_vector_embedding)
    print(f"Cosine similarity: {cosine:.4f}")
    ...

def find_similar_words(target_word, word_to_idx, embeddings, top_n=5):
    """Find words most similar to the target word based on cosine similarity."""
    if target_word not in word_to_idx:
        raise ValueError(f"Word '{target_word}' not in vocabulary.")

    target_idx = word_to_idx[target_word]
    target_vector = embeddings[target_idx]

    similarities = {}
    for word, idx in word_to_idx.items():
        if word != target_word:
            similarity = cosine_similarity(target_vector, embeddings[idx])
            similarities[word] = similarity

    # Sort words by similarity in descending order
    sorted_similar_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similar_words[:top_n]

def Solve():
    raw_text = load_file()
    stop_word = load_stop_word_data()
    words = separate_sentences_and_words(raw_text, stop_word)
    
    m = make_dictionary(words)

    training_data = make_training_data(words= words, m= m)
    try:
        w1 = pd.read_csv("w1.csv", header= None).values
        w2 = pd.read_csv("w2.csv", header= None).values
    except Exception as e:
        w1, w2 = initialization_embedding_matrix(m= m)  
        
    w1, w2 = train(w1, w2, m= m, training_data= training_data)
    pd.DataFrame(w1).to_csv("w1.csv", index= False, header= False)
    pd.DataFrame(w2).to_csv("w2.csv", index= False, header= False)  
    
    while(True):
        word_similarity(w1, m= m)
        s = input("Tiếp tục? Y=Yes : N=No ")
        if s == "Y" or s == "y":
            break
    
    # tw = ViTokenizer.tokenize("thạch lam")
    # r = find_similar_words(tw, word_to_index, w1, top_n= 5)
    # print(f"Words similar to '{tw}':", r)


def main():
    Solve()
    
if __name__ == "__main__":
    main()