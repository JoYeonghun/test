# main.py
import streamlit as st

## BERT
import urllib.request
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

st.text('hello Streamlit!')

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

## 경고 해제
pd.set_option('mode.chained_assignment',  None)

train_data = pd.read_csv('https://raw.githubusercontent.com/JoYeonghun/test/main/data/Sentence(100)_Embedding.csv')
t_data = train_data[['Q','A','embedding']]

## t_data['embedding'] str -> numpy 형변환
a = []
for _ in range(100):
  tmp = t_data['embedding'][_].replace('[', '').replace(']', '').replace('\n', '')
  s_to_n = np.fromstring(tmp, dtype='f', sep=' ')
  a.append(np.array(s_to_n, dtype='f'))
  if _ % 10 == 0 :
    print("\r", end='')
    print(f'{_ // 10 * 10}% :', '##'*(_ // 10), end='')

new_data = t_data[['Q','A']]
new_data['embedding'] = a

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_similar_answer(input):
    embedding = model.encode(input)
    new_data['score'] = new_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return new_data.loc[new_data['score'].idxmax()]['A']

## Text Input
st.text("안녕하세요 00님! 오늘의 일기를 작성해주세요")
message = st.text_area("일기 작성 칸")
st.text("기록이 완료됐습니다.")

## 위로 문장 출력
st.write(return_similar_answer(message))
