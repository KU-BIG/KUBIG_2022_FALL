## Deep Learning NLP Project

#### Team : 15기 김지호, 신윤, 염윤석, 우명진, 최민경
### 데이콘 뉴스 토픽 분류 AI 경진대회
#### link : https://dacon.io/competitions/official/235747/overview/description

"한국어 뉴스 헤드라인을 이용하여, 뉴스의 주제를 분류하는 알고리즘 개발"

### **0. DL library : pytorch & huggingface**

### **1. DATA 구성**
   * index : 헤드라인 인덱스
   * title : 뉴스 헤드라인
   * topic_idx : 뉴스 주제 인덱스 값(label)
    
### **2. DATA Preprocessing**
  * 품사태깅
  * 숫자, 특수문자 제거 + 한 글자 미만 제거
  * 문장부호 제거
  * 영어/한자 --> 한글로 변환
  * 추가 전처리 : 이상문자열 제거

### **3. Modeling**
  * Roberta-small
  * Roberta-base
  * Roberta-large
  * Koelectra-base


### **RESULT : Accuracy**
**Roberta -large** 
    
    : public 114 ACC :0.85454

**Voting Ensemble** : koelectra(0.1) + large(0.3) + small(0.5) + base(0.1) 

    : public 145 ACC :0.84030
