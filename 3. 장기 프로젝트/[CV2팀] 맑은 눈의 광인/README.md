# 👀 양재 허브 인공지능 오픈소스 경진대회 👀
<img width="60%" src="https://user-images.githubusercontent.com/97013710/210364441-89d27d3f-e22e-4156-ad14-b1a73665dd46.jpeg">
맑은 눈의 광인이 되어 세상을 바라보자! 


Team : 15기 장수혁, 염윤석, 최민경, 16기 박민규

<br/><br/>

## Project Descriptions

**Link** : [AI 양재 허브 인공지능 오픈소스 경진대회 (DACON)](https://dacon.io/competitions/official/235977/overview/description)

**[주제]**  
이미지 초해상화(Image Sper-Resolution)를 위한 AI 알고리즘 개발  

**[배경]**  
오픈 소스 이미지 데이터를 활용하여 인공지능 컴퓨터 비전의 '이미지 초해상화' 분야 연구개발  

**[설명]**  
품질이 저하된 저해상도 촬영 이미지(512x512)를 고품질의 고해상도 촬영 이미지(2048x2048)로 생성  

**[평가 산식]**  
PSNR(Peak Signal-to-Noise Ratio) = $10log_{10}(R^2/MSE)$
-	생성 혹은 압축된 영상의 화질에 대한 “손실 정보”를 평가
-	손실이 적을수록(=화질이 좋을 수록) 높은 값  

<br/><br/>

## 👣 Score(Public)
RRDB+ : 23.40812(38th)

<br/><br/>

## 🌐 Environment
Colab Pro+  
GPU: A100-SXM4-40GB * 1(Main) , Tesla T4*1(Sub)

<br/><br/>

## 🔥 Competition Strategies

**1. Patches(for data augmentation)**  
- Train patches : original 1640 images → 26240(1640*16) patches (X4 downsampling, non-overlapping)  
- Test patches: original 18 images → 882(18*49) patches (X4 downsampling, overlapping(to remove border artifacts)) 

<br/>

**2. Data Transform**  
Non-destructive transformations (not to add or lose the information)
- Flip  
- Transpose  
- RandomRotate  
- ShiftScaleRotate  

<br/>

**3. Training Methods**
- EarlyStopping  
  > To prevent overfitting  
  > If validation loss does not improve after given patience(=2), training is earlystopped  
- Fine-tuning with pre-trained model  
  > pretrained model : [RRDB_PSNR_x4.pth](https://github.com/xinntao/ESRGAN/tree/master/models)(the PSNR-oriented model with high PSNR performance)  
  > Retraining entire model : Judging that the similarity between DF2K dataset(pretrained model) and our training datset is small  

<br/>

**4. Loss Function**
- L1 loss + L2 loss (2:1)
  > L2 loss : PSNR is based on MSE  
  > L1 loss: For better convergence [https://arxiv.org/pdf/1707.02921v1.pdf](https://arxiv.org/pdf/1707.02921v1.pdf)

<br/>

**5. Learning Scheduler, Optimizer**
- StepLR  
  > step_size = 3, gamma = 0.5  
  > Decays the learning rate of each parameter in half once per 3 epochs 
- Adam  

<br/>

**6. Post Processing**
- Geometric Self-Ensemble [https://arxiv.org/pdf/1707.02921v1.pdf](https://arxiv.org/pdf/1707.02921v1.pdf)
  > <img width="751" alt="KakaoTalk_Photo_2023-01-04-11-58-49" src="https://user-images.githubusercontent.com/55012723/210476203-015eac00-d0e0-4d10-8eb5-a772a9910097.png">


<br/><br/> 

## Main configuration & Hyperparameters
'''
1. Manuel_seed : 42  

2. Model :  
   > num_feat : 64 , Channel number of intermediate features.  
   > growth_channel: 32 , Channels for each growth(dense connection).  
   > num_block: 23 , number of RRDB blocks.  

3. Dataloader :  
   > train_batch_size : 4  
   > test_batch_size: 1  
   > num_workers: 4   

4. Train :  
   > epochs: 7  
   > optim_g: {type: Adam, lr: 1e-4, betas: [0.9, 0.99]}  

'''

<br/><br/>

## Code Descriptions
1. DACON_AISR_TRIAL
- EDSR, SRGAN, SWINIR


2. DACON_AISR_BEST
- RRDB, RRDB+(Self-ensemble)
