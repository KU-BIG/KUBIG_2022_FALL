CLIP-OpenAI Analysis and Application
-----------------------
##### Team : 15기 김진수, 이제윤, 박지우 16기 엄기영

### CLIP ::
![image](https://user-images.githubusercontent.com/87689944/187473772-32fe17bf-61d5-4805-87dd-9db9e181b7a1.png)

기존의 state of art computer vision 모델은 지도학습 기반으로, 일반성과 활용성에 제약이 많으나,
CLIP의 경우 라벨이 아닌, 자연어 캡션을 통해 학습(natural language supervision)함으로써 much broader source of supervision을 leverage할 수 있는 모델입니다.


CLIP 논문 [Learning Transferable Visual Models From Natural Language Supervision]

논문 링크 : https://arxiv.org/abs/2103.00020


논문의 내용을 토대로, CLIP모델을 공부해보고, 이를 여러 방면으로 활용하는 프로젝트를 진행하였습니다.

------------------------
### 1. CLIP 모델을 pytorch 데이터 및 다양한 데이터셋에 적용
  + CIFAR100, Food101등 파이토치 데이터셋에 적용
  + 논문에서 등장한 Zero-shot Learning 성능 비교를 확인
### 2. AI 편향성 확인
  + CLIP의 경우, 인터넷 상의 데이터셋에 대해 bias를 고려하지 않고 학습을 진행, 논문에 데이터셋에 대한 편향성이 언급되어 있음
  + Gender Classification Dataset 을 이용해 Gender, Races 에 따른 분류 성능 차이 및 편향성을 확인
  + https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
### 3. grad-CAM 적용
  + ExplainableAI인 grad-CAM을 활용해 CLIP의 이미지 분류 모델이 이미지의 어떤 부분에 가중치를 크게 두었는지 확인
### 4. Zero-shot Object Detection 적용

--------------------------
## Reference
https://greeksharifa.github.io/computer%20vision/2021/12/19/CLIP/

https://velog.io/@dongdori/CLIP2021-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
