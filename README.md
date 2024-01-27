# 자연스러운 번역문 생성
"AI Connect 모의경진대회" 

보다 자연스러운 기계 번역문을 생성하기 위한 영한번역 및 사후교정 문제

**주최**: 중소벤처기업진흥공단

**주관**: 마인즈앤컴퍼니 (AI Connect)

<br>

## OverView

### 대회소개:

보다 자연스러운 기계 번역문을 생성하기 위한 영한번역 및 사후교정 문제

**제한사항**: 외부데이터 사용불가

**참여기간**: 2023.10.30 ~ 11.03

**참여방식**: 개인

<br>

## 데이터

input : 영어 문장 (7만여건)

output : 한국어 문장 (1만여건)

### 데이터 설명

**train.csv**

<p align='center'><img src="assets/src03.png" width="720"></p>

<p align='center'><img src="assets/src04.png" width="720"></p>

**test.csv**

<p align='center'><img src="assets/src05.png" width="380"></p>

**sample_submission.csv** 

<p align='center'><img src="assets/src06.png" width="380"></p>

<br>

## 평가지표

- 평가지표: BLEU Score (sacrebleu, Bi-Lingual Evaluation Understudy)

<br>

$$ \textrm{Unigram Precision} \ P = \frac{m}{w_t} \ \ \ ⟶ \textrm{n-gram 정밀도} $$

<br>

$$ \textrm{where} 
\begin{cases} 
m: \textrm{number of tokens both in reference and predition} \\ 
w_t: \textrm{number of tokens in prediction} 
\end{cases} 
$$

<br>

$$ 
\textrm{Brevity penalty }p 
\begin{cases} 
1 \ \ \ \ \ \ \ \ \ \ \ \textrm{ if } \ c > r  \\
e^{(1-\frac{r}{c})} \ \ \ \textrm{ if } \ c ≦ r 
\end{cases} 
⟶ \textrm{길이 패널티} 
$$  

<br>

$$ ⟹ \textrm{BLUE} = p⋅e^{\Sigma^n_{n=1}(\frac{1}{N}\log⋅P_n)}\textrm{ where } N = 4 $$

<br>

- 테스트 데이터 분할
<p align='center'><img src="assets/src02.png" width="420"></p>

<br>

## 탐색적 데이터분석 EDA

## 피쳐엔지니어링

## 모델선정 및 훈련

## 한계 점

## 결과
