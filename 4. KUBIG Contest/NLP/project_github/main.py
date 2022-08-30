import datetime
from datetime import timedelta
import schedule
import time

from src.newsletter import PyMail
from src.newsletter import make_final_contents
from config import Config

import math
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from transformers import BertModel, BertTokenizer, AdamW, PreTrainedTokenizerFast, BartModel

from torch.nn.init import xavier_uniform_

import pytorch_lightning as pl

import kss
import re

from model import SummDataset, PositionalEncoding, TransformerEncoderLayer, ExtTransformerEncoder, PositionwiseFeedForward, MultiHeadedAttention, Summarizer

# G메일 계정 정보 초기화
c = Config()
address = c.GMAIL_ACCOUNT['address']
password = c.GMAIL_ACCOUNT['password']

# 뉴스 날짜 정의 (오늘 날짜)
search_word_list = datetime.today()

def send_mail_func():
    """
    컨텐츠 생성 및 이메일 발송 기능 호출 함수
    """

    # 컨텐츠 생성
    contents = make_final_contents(search_word_list)
    # 타이틀 및 컨텐츠 작성
    #date_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y년 %m월 %d일')
    date_str = datetime.datetime.strftime(datetime.datetime.now() - timedelta(1),'%Y년 %m월 %d일')
    title = f"""📢 KUBIG 금융뉴스요약 알림서비스 ({date_str}) 📢"""
    contents=f'''{contents}'''
    # # 첨부파일 경로 설정
    # attachment_path = f"D:/Task.txt"
    # 수신자 정보 설정
    target_email_id = "njj7991@gmail.com"
    # 문서 타입 설정 - plain, html 등
    subtype = 'html'
    # 세션 설정
    PM = PyMail(address, password)
    # 메일 발송
    PM.send_mail(target_email_id, title, contents, subtype)
    print("발송 완료")

# 스케줄 등록
# schedule.every(60).minutes.do(send_mail_func)
schedule.every().day.at("23:59").do(send_mail_func)

while True:
    schedule.run_pending()
    time.sleep(1)