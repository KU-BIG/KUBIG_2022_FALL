## 한문철 내과의원

___________________________________

### 차량 이상현상 탐지 프로젝트

+ 주제 : 블랙박스 영상 등을 활용해, 차량이 현재 처해있는 위험도를 구하고 이를 바탕으로 사용자에게 신호를 주는 시스템 개발

![image](https://user-images.githubusercontent.com/87689944/212276275-d81e6ec8-5148-4f42-b2a2-2035923a53e5.png)

도로주행 영상 데이터로 Yolo v7 모델 학습 진행

#### [모델 구조]

![image](https://user-images.githubusercontent.com/87689944/212276514-ea13c4d9-6b47-4ecb-a114-fa2cb1a0a9da.png)

#### [위험도 함수]

![image](https://user-images.githubusercontent.com/87689944/212276618-c97ceefa-4054-43d8-828a-39bf411d35fe.png)

+ 위험구간 Bounding Box (녹색 상자) 부분과 물체와의 intersection을 지표로 사용. 

+ Yolov7 모델 특징상 물체가 가까워지면 인식 못하는 현상 발생

+ 이를 바탕으로 위험도 값이 갑자기 하락 하는 경우를 위험 상황으로 지정함.

![image](https://user-images.githubusercontent.com/87689944/212276818-dec7b31c-e1e5-4242-a479-e62cce829c08.png)

_______________________________

### 유방암의 임파선 전이 예측 프로젝트 [Dacon]

주제 : 유방암 전력이 있는 환자의 정보를 활용, 임파선 전이 여부를 미리 예측하는 시스템 개발

#### [모델 구조]

![image](https://user-images.githubusercontent.com/87689944/212277013-6b28886c-6ade-45c1-b042-d7e71aa354e4.png)

#### [Result]

Model Score : 0.714

Best model Score : 0.751



