# 研究のメモ

## 比較用論文

### タイトル
Application of the Generic Feature Selection Measure in Detection of Web Attacks

### URL
https://pdfs.semanticscholar.org/ab9e/74875aa28d26c6871e670ad6f38391ec929e.pdf

### 備考
#### 用いている特徴量
- Length of the request
- Lenght of the arguments
- Number of arguments
- Number of digits in the arguments
- Number of digits in the path
- Length of the path
- Number of letters in the arguments
- Number of 'special' char in the path
- Number of letters char in the path
- Number of 'special' char in the path
- Maximum byte value in the request

### 結果
||C4.5|CART|RandomTree|RandomForest|
|---|---|---|---|---|
|Detection Rate|0.9449|0.9412|0.9270|0.9368|
|False Positive Rate|0.59|0.62|0.78|0.72|

(おそらく Detection Rate = Recall)

---

### タイトル
Web Application Firewall using Character-level Convolutional Neural Network

### URL
https://pdfs.semanticscholar.org/ab9e/74875aa28d26c6871e670ad6f38391ec929e.pdf (要旨)

(本文も読みたいけれどIEEE会員ではないため不可能らしい)

### 備考
- テキスト中の文字符号を畳み込みニューラルネットワークを用いて学習し、正常・異常の2クラス分類をすることで検知を行っている
- IEEE CSPA 2018 Best Paper Award
- > state-of-the-art（従来研究の最善の結果：82％） ←意味不明
- > 高速に動作(2.35msec/件) ←比較不能

### 結果
Detection accuracy: 0.988
(PrecisionやRecall, False Positive Rateも見てみたいけれど・・)
