import re
import gzip
import imp
import time
import urllib.parse
import numpy as np

import pickle

from gevent.server import StreamServer
from mprpc import RPCServer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


ascii_symbols = r'[ -/:-@[-`{-~]' # asciiコード中の記号(スペース含む)とマッチする正規表現
ascii_control_chars = r'[\x00-\x1f]|\x7f' # asciiコード中の制御文字とマッチする正規表現
reg = re.compile('({}|{})'.format(ascii_symbols, ascii_control_chars))
def split_text(str):
    str = str.lower() # 大文字をすべて小文字に変換
    return reg.split(str)

# データセットから1件ずつHTTPテキストをパースして取得
def dataset2texts(filepath):
    text = ''
    with gzip.open(filepath, mode='rt') as f:
        for line in f:
            if ('GET' in line) or ('POST' in line) or ('PUT' in line):
                if text != '':
                    yield text
                    text = ''
            text = text + line
        yield text

# HTTPテキストを単語配列に変換
def text2words(str):
    arr = str.split('\n')
    method, url, _ = arr[0].split(' ')
    u = urllib.parse.urlparse(url)

    param = ''
    if method == 'GET':
        param = u.query
    elif method == 'POST' or method == 'PUT':
        for line in reversed(arr):
            if line == '':
                continue
            else:
                param = line
                break

    param = urllib.parse.unquote_plus(param) # decoding

    words = split_text(u.path) + split_text(param)
    return words


class ClfServer(RPCServer):
    def __init__(self):
        norm_train = list(dataset2texts('./static/original/normalTrafficTraining.txt.gz'))
        anom_test = list(dataset2texts('./static/original/anomalousTrafficTest.txt.gz'))

        X = np.array(norm_train + anom_test)
        y = np.array(['norm'] * len(norm_train) + ['anom'] * len(anom_test))

        # BoWベクトライザの生成
        self.vectorizer = CountVectorizer(analyzer=text2words)
        self.vectorizer.fit(X)

        # ベクトル生成
        X_train = vectorizer.transform(X)

        # ランダムフォレスト分類器の生成
        self.clf = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=6)
        self.clf.fit(X_train, y)

    def predict(text):
        vec = self.vectorizer.transform([text])
        pred = self.clf.predict(x)[0]

        # return 'norm' or 'anom'
        return pred # predには何が入ってるのかチェック


if __name__ == '__main__':
    server = StreamServer(('127.0.0.1', 6000), ClfServer())
    server.serve_forever()
