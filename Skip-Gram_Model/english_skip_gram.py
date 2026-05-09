import re
import string

import nltk
from nltk import sent_tokenize, word_tokenize

import numpy as np
from numpy.linalg import norm

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

FILE_PATH = "english_corpus.txt"
WINDOW_SIZE = 2
LEARNING_RATE = 0.01
EPOCH = 1000
EMBEDDING_SIZE = 300
 
def load_file(url:str) -> str:
    with open(url, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def text_preprocessing(text:str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def separate_sentences_and_words(text:str) -> list:
    sentences = sent_tokenize(text)
    sentences_preprocessed = [text_preprocessing(sentence) for sentence in sentences]
    words = [word_tokenize(sentence) for sentence in sentences_preprocessed]
    return words
    
def make_dictionary(words: list) -> tuple[dict, dict]:
    wti = dict() # word to index
    itw = dict() # index to word
    index = 0
    for word in words:
        for w in word:
            if w in wti:
                continue
            wti[w] = index
            itw[index] = w
            index += 1
    return wti, itw
    

def make_training_data(words: list, word_to_index:dict , window_size:int = WINDOW_SIZE)->list:
    td = list()
    for word in words:
        length = len(word)
        print(word)
        for index in range(length):
            for j in range(max(index-window_size, 0), min(window_size+index+1, length)):
                if j == index :
                    continue
                td.append((word_to_index[word[index]], word_to_index[word[j]]))
    
    return td
    ...    
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis = 0)
    ...    

def initialization_embedding_matrix(word_in_index: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(word_in_index)
    
    w1 = np.random.rand(n, EMBEDDING_SIZE)
    w2 = np.random.rand(EMBEDDING_SIZE, n)
    return w1, w2
    ...

def one_hot_vector(index: int, range: int):
    v = np.zeros(range)
    v[index] = 1
    return v

def train(w1, w2, word_to_index: dict, training_data: list) -> np.array:
    total_words = len(word_to_index)
    print(f"W1: {w1.shape}")
    print(f"W2: {w2.shape}")
    for epoch in range(1,EPOCH+1):
        loss = 0
        for center_word, context_word in training_data:
            
            y_center_word = one_hot_vector(center_word, total_words)
            y_context_word = one_hot_vector(context_word, total_words)
            
            h = np.dot(y_center_word,w1)
            u = np.dot(h, w2)
            y_pred = softmax(u)
            
            loss -= np.log(y_pred[context_word])
            
            error = y_pred - y_context_word
            dL_dh = np.outer(y_center_word, np.dot(w2, error))
            dL_dw2 = np.outer(h, error)
            
            w1 = w1 - LEARNING_RATE*dL_dh
            w2 = w2 - LEARNING_RATE*dL_dw2
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, loss: {loss:.4f}")
            
    return w1, w2
    ...

def cosine_similarity(arr1, arr2):
    return np.dot(arr1, arr2)/(norm(arr1)*norm(arr2))
    ...

def word_similarity(w1 , word_to_index: dict, index_to_word: dict):
    a = input("Nhập từ thứ nhất: ").strip()
    b = input("Nhập từ thứ hai: ").strip()
    if a not in word_to_index:
        print(f"{a} không có từ điển")
        return
    if b not in word_to_index:
        print(f"{b} không có từ điển")
        return
    
    a_vector_embedding = w1[word_to_index[a]]
    b_vector_embedding = w1[word_to_index[b]]
    
    # print(f"{a} embedding vector: {a_vector_embedding}")
    # print(f"{b} embedding vector: {b_vector_embedding}")
    
    cosine = cosine_similarity(a_vector_embedding, b_vector_embedding)
    print(f"Cosine similarity: {cosine:.4f}")
    ...

def plot_word_similarity(w1, word_to_index: dict, words):
    vectors = []
    valid_words = []

    for key,value in word_to_index.items():
        vectors.append(w1[word_to_index[key]])
        valid_words.append(key)
        
    if len(vectors) < 2:
        print("Ko đủ ngữ liệu")
        return
    
    vectors = np.array(vectors)
    cosine_similarity_matrix = np.dot(vectors, vectors.T)/ (
        np.linalg.norm(vectors, axis= 1)[:, None] * np.linalg.norm(vectors, axis= 1)
    )
    
    plt.figure(figsize=(30,28))
    sns.heatmap(
        cosine_similarity_matrix,
        xticklabels= valid_words,
        yticklabels= valid_words,
        cmap= "coolwarm",
        annot= True,
        fmt= ".2f"
    )
    plt.title("Biểu đồ tương quan giữa các từ (Cosine Similarity)")
    plt.xlabel("Từ")
    plt.ylabel("Từ")
    plt.savefig("chart.png", dpi = 300, bbox_inches="tight")
    # plt.show()
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
    text = load_file(FILE_PATH)
    words = separate_sentences_and_words(text= text)
    word_to_index, index_to_word = make_dictionary(words= words)
    print(word_to_index)
    training_data = make_training_data(words= words, word_to_index= word_to_index)
    # print(len(training_data))
    w1, w2 = initialization_embedding_matrix(word_in_index= word_to_index)
    
    # print(training_data)

    w1, w2 = train(w1, w2, word_to_index= word_to_index, training_data= training_data)
    
    plot_word_similarity(w1= w1, word_to_index= word_to_index, words= words)
    
    # while(True):
    #     word_similarity(w1, word_to_index= word_to_index, index_to_word= index_to_word)
    #     s = input("End?Y=Yes:N=No ")
    #     if s == "Y":
    #         break
    
    tw = "dog"
    result = find_similar_words(tw, word_to_index, embeddings= w1, top_n= 5)
    print(f"Words similar to '{tw}':", result)
    ...     

    
def main():
    Solve()
    ... 
    
if __name__ == "__main__":
    main()