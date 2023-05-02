import csv 
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

stopwords = ['사랑', '우리', '마음'] # step 3에서 추천된 불용어
csvs = ["발라드", "랩힙합", "트로트", "CCM"]

def csv_data(lists) :
    result = [[] for i in range(len(lists))]
    f = open('2_Origin Dataset.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        if line[-1] == 'genre' : continue
        result[int(csvs.index(line[-1]))].append(line[2])
    f.close()
    return result

def replace_1(lists) : # '\n'제거
    result = []
    for l in lists :
        c = []
        for v in l :
            c.append(v.replace("\n", " "))
        result.append(c)
    return result

def replace_2(lists) : # 한글 빼고 모두 제거
    result = []
    for l in lists :
        c = []
        for v in l :
            c.append(re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]',"", v))
        result.append(c)
    return result

def del_null(lists) : # 빈 문자열 제거
    result = []
    for li in lists :
        array = []
        res = []
        for l in li :
            res.append(l)
            array.append(l.replace(' ', ''))
        delete_ly = [i for i, sentence in enumerate(array) if len(sentence) < 1]
        for l in delete_ly[::-1] :
            res.pop(l)
        result.append(res)
        
    lens = []
    for l in result :
        lens.append(len(l))
    range_c = min(lens)
    return result, range_c

def del_stopword(lists, stopwords) : # 불용어 제거
    result = []
    for l in lists :
        c = []
        for v in l :
            for word in stopwords :
                v = v.replace(word, "")
            c.append(v)
        result.append(c)
    return result

def del_repetition(lists) : # 같은 장르 내에서 중복 제거
    # 다른 장르에서 겹치는 곡이 있어도 이는 없애지 않는 것이 좋다고 판단하였습니다.
    origin_len = 0
    lists_2 = [] # 같은 장르 내 중복 제거 완료된 리스트
    for l in lists :
        origin_len += len(l)
        changed_list = []
        for v in l :
            if v not in changed_list :
                changed_list.append(v)
        lists_2.append(changed_list)
        
    changed_len = 0
    for l in lists_2 :
        changed_len += len(l)
    return lists_2

okt = Okt()
def get_okt(lists, n, mode) :
    pp = []
    pp_test = []
    x_train = []
    x_test = []
    if mode == "train" :
        for i, l in enumerate(lists) :
            for sentence in l[:n]:
                tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
                pp.append((tokenized_sentence, i))
                x_train.append(tokenized_sentence)
        return x_train, pp
    elif mode == "test" :
        for i, l in enumerate(lists) :
            for sentence in l[n:]:
                tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
                pp_test.append((tokenized_sentence, i))
                x_test.append(tokenized_sentence)
        return x_test, pp_test

def processing(kk) :
    data_b = csv_data(csvs)
    data_c = replace_1(data_b)
    data_d = replace_2(data_c)
    data_e = del_stopword(data_d, stopwords)
    data_f = del_repetition(data_e)
    data_g, range_c = del_null(data_f)
    x_data = []
    for line in data_g :
        x_data.append(line[:range_c])
    X_train, pp = get_okt(x_data, int(range_c*0.8), "train")
    
    tokenizer = Tokenizer(kk)
    tokenizer.fit_on_texts(X_train)
    
    return tokenizer
    
processing(1415)