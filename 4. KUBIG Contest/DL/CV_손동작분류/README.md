# 2021 Ego-vision 손동작 인식 AI 경진대회
Ego-Vision 관점의 영상에서 추출한 이미지 학습데이터를 활용한 인공지능 모델 기반의 손동작 인식 및 분류 모델 개발

## 데이터
1. Train set: 649개의 폴더, 5888개의 이미지 (+ json파일: keypoint 좌표, label 등)
2. Test set: 217개의 폴더, 2038개의 이미지
3. Hand_gesture_pose(csv): 이미지 라벨에 따른 손동작 설명이 들어있는 파일
4. sample submission(csv): 제출파일

## 전처리
1. crop & keypoints 제거
2. Augmentation(mix-up & Albumentations)

## 모델
1. ResNet50, EfficientNet
2. Label Smoothing loss
+post-processing 진행
