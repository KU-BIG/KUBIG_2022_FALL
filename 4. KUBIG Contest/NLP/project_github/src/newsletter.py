import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import codecs
import logging
import requests
from bs4 import BeautifulSoup
from pretty_html_table import build_table

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time

from selenium import webdriver

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

################################################################
################################################################
################################################################

class SummDataset(Dataset):

    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: PreTrainedTokenizerFast, 
        max_token_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        tokenlist = []
        for sent in data_row.article_original:
            tokenlist.append(tokenizer(
                text = sent,
                add_special_tokens = True)) #, # Add '[CLS]' and '[SEP]'
    
        src = [] # 토크나이징 된 전체 문단
        labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
        segs = []  #각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
        clss = []  #[CLS]토큰의 포지션값을 지정

        odd = 0
        for tkns in tokenlist:
            if odd > 1 : odd = 0
            clss = clss + [len(src)]
            src = src + tkns['input_ids']
            segs = segs + [odd] * len(tkns['input_ids'])
            if tokenlist.index(tkns) in data_row.extractive :
                labels = labels + [1]
            else:
                labels = labels + [0]
            odd += 1
        
            #truncation
            if len(src) == MAX_TOKEN_COUNT:
                break
            elif len(src) > MAX_TOKEN_COUNT:
                src = src[:self.max_token_len - 1] + [src[-1]]
                segs = segs[:self.max_token_len]
                break
    
        #padding
        if len(src) < MAX_TOKEN_COUNT:
            src = src + [0]*(self.max_token_len - len(src))
            segs = segs + [0]*(self.max_token_len - len(segs))
            
        if len(clss) < MAX_TOKEN_COUNT:
            clss = clss + [-1]*(self.max_token_len - len(clss))
        if len(labels) < MAX_TOKEN_COUNT:
            labels = labels + [0]*(self.max_token_len - len(labels))

        return dict(
            src = torch.tensor(src),
            segs = torch.tensor(segs),
            clss = torch.tensor(clss),
            labels= torch.FloatTensor(labels)
        )

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class ExtTransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, d_ff=2048, heads=8, dropout=0.3, num_inter_layers=2): 
        super(ExtTransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, heads, d_ff, dropout)
            for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.wo = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask) 

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)
        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18) # how can i fix it to use fp16...

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

class Summarizer(pl.LightningModule): #

    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.max_pos = 512
        self.bert = BertModel.from_pretrained('gogamza/kobart-base-v1') #, return_dict=True)
        self.ext_layer = ExtTransformerEncoder()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss = nn.BCELoss(reduction='none')
    
        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, segs, clss, labels=None): #, input_ids, attention_mask, labels=None):
        
        mask_src = ~(src == 0) #1 - (src == 0)
        mask_cls = ~(clss == -1) #1 - (clss == -1)

        top_vec = self.bert(src, token_type_ids=segs, attention_mask=mask_src)
        top_vec = top_vec.last_hidden_state
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        
        loss = 0
        if labels is not None:
            loss = self.loss(sent_scores, labels)
            
            loss = (loss * mask_cls.float()).sum() / len(labels)
        
        return loss, sent_scores

    def step(self, batch):

        src = batch['src']
        if len(batch['labels']) > 0 :
            labels = batch['labels']
        else:
            labels = None
        segs = batch['segs']
        clss = batch['clss']
        
        loss, sent_scores = self(src, segs, clss, labels)    
        
        return loss, sent_scores, labels


################################################################
################################################################
################################################################


class PyMail:
    """
    Python Email 전송 클래스
    """
    def __init__(self, my_email_id, my_email_pw):
        """
        G메일 계정, SMTP 정보 및 세션 초기화
        """
        # 계정 정보 초기화
        self.my_email_id = my_email_id
        self.my_email_pw = my_email_pw
        # G메일 SMTP 호스트, 포트 정보 초기화
        self.smtp_host = 'smtp.gmail.com'
        self.smtp_port = 587
        # 세션 정의
        self.session = smtplib.SMTP(self.smtp_host, self.smtp_port)

    def send_mail(self, target_email_id, title, contents, subtype=None, attachment_path=None):
        """
        이메일 전송 메서드
        - 수신자 이메일, 제목, 내용, 문서타입, 첨부 파일 경로
        """
        # 세션 보안 TLS 시작
        self.session.starttls()
        # 세션 계정 로그인
        self.session.login(self.my_email_id, self.my_email_pw)
        # 제목, 본문 작성
        msg = MIMEMultipart()
        msg['Subject'] = title
        if not subtype:
            msg.attach(MIMEText(contents, 'plain'))
        else:
            msg.attach(MIMEText(contents, subtype))
        # 파일첨부 (파일 미첨부시 생략가능)
        if attachment_path:
            fileName = attachment_path.split("/")[-1]
            attachment = open(attachment_path, 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= " + fileName)
            msg.attach(part)
        # 메일 전송
        try:
            self.session.sendmail(self.my_email_id, target_email_id, msg.as_string())
            self.session.quit()
        except:
            self.session.quit()

def df_to_html_table(df, index=False):
    """
    Pandas DataFrame을 HTML 테이블 태그로 변환
    """
    return build_table(df, 'blue_light')


def data_process(text):
    
    MAX_TOKEN_COUNT = 512
    N_EPOCHS = 3
    BATCH_SIZE = 2

    # 문장 분리 하고,
    text = re.sub('\[.*\]|\s-\s.*', '', text)
    sents = kss.split_sentences(text)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

    #데이터 가공하고,
    tokenlist = []
    for sent in sents:
        tokenlist.append(tokenizer(
            text = sent,
            add_special_tokens = True)) #, # Add '[CLS]' and '[SEP]'

    src = [] # 토크나이징 된 전체 문단
    labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
    segs = []  #각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
    clss = []  #[CLS]토큰의 포지션값을 지정

    odd = 0

    for tkns in tokenlist:

        if odd > 1 : odd = 0
        clss = clss + [len(src)]
        src = src + tkns['input_ids']
        segs = segs + [odd] * len(tkns['input_ids'])
        odd += 1

        #truncation
        if len(src) == MAX_TOKEN_COUNT:
            break
        elif len(src) > MAX_TOKEN_COUNT:
            src = src[:MAX_TOKEN_COUNT - 1] + [src[-1]]
            segs = segs[:MAX_TOKEN_COUNT]
            break

    #padding
    if len(src) < MAX_TOKEN_COUNT:
        src = src + [0]*(MAX_TOKEN_COUNT - len(src))
        segs = segs + [0]*(MAX_TOKEN_COUNT - len(segs))

    if len(clss) < MAX_TOKEN_COUNT:
        clss = clss + [-1]*(MAX_TOKEN_COUNT - len(clss))

    return dict(
        sents = sents, #정답 출력을 위해...
        src = torch.tensor(src),
        segs = torch.tensor(segs),
        clss = torch.tensor(clss),
    )

def summarize_test(text):
    data = data_process(text.replace('\n',''))

    # 저장된 모델 불러오기
    trained_model = torch.load('C:/Users/njh20/Desktop/project/model_fulldata2.pt') 
    
    #trained_model에 넣어 결과값 반환
    _, rtn = trained_model(data['src'].unsqueeze(0), data['segs'].unsqueeze(0), data['clss'].unsqueeze(0))
    rtn = rtn.squeeze()
    
    # 예측 결과값을 받기 위한 프로세스
    rtn_sort, idx = rtn.sort(descending = True)
    
    rtn_sort = rtn_sort.tolist()
    idx = idx.tolist()

    end_idx = rtn_sort.index(0)

    rtn_sort = rtn_sort[:end_idx]
    idx = idx[:end_idx]
    
    if len(idx) > 3:
        rslt = idx[:3]
    else:
        rslt = idx
        
    summ = []
    print(' ')
    for i, r in enumerate(rslt):
        summ.append(data['sents'][r])
        #print(summ[i])

    return summ

    

def make_contents(search_word_list):
    """
    웹 크롤링 함수
    """

    df = pd.DataFrame()
    #search_word = (datetime.today() - timedelta(1)).strftime("%Y-%m-%d") # 어제 날짜
    search_word = '2022-08-26'

    # 해당 url의 html문서를 soup 객체로 저장
    url = f'https://finance.naver.com/news/mainnews.naver?date={search_word}'

    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.select('div.mainNewsList > ul.newsList > li.block1 > dl > dd.articleSubject > a')
    #source = soup.select("div.mainNewsList > ul.newsList > li.block1 > dl > dd.articleSummary > span.press")
    link = soup.select('div.mainNewsList > ul.newsList > li.block1 > dl > dd.articleSubject > a')

    title_list = list(map(lambda x: x.text, title))
    link_list = list(map(lambda x: 'https://finance.naver.com' + x.attrs['href'], link))

    text_list = []

    for url in link_list:
        wd = webdriver.Chrome('chromedriver.exe')
        wd.get(url)
        req = wd.page_source
        soup = BeautifulSoup(req, 'html.parser')

        ul =soup.find('div',{"class":"articleCont"}).text
        cleantext = BeautifulSoup(ul, "lxml").text
        text_list.append(cleantext)

    tmp = pd.DataFrame({"Title": title_list, "Link": link_list, "Text": text_list})
    df = df.append(tmp)

    df = df[['Title', 'Link', 'Text']]
    df.index = range(len(df))

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

    # 저장된 모델 불러오기
    trained_model = torch.load('C:/Users/njh20/Desktop/project/model_fulldata2.pt') 

    MAX_TOKEN_COUNT = 512
    N_EPOCHS = 3
    BATCH_SIZE = 2

    sentence = []
    for news_sentence in df['Text'].values:
        sentence.append(news_sentence)
    
    array = []
    for i in sentence:
        sum_list = summarize_test(i)
        sum = " ".join(sum_list)
        array.append(sum)

    df['Summarize'] = array

    return df


def preprocessing(df):
    """
    전처리 함수
    """
    new_title_list = []

    for idx, row in df.iterrows():

        title = row['Title']
        link = row['Link']
        
        new_title = f"""<a href="{link}">{title}</a>"""

        new_title_list.append(new_title)

    df['Title_Link'] = new_title_list

    # 결과
    df_cls = df[['Title_Link','Summarize']]
    colDict = {"Title_Link": "제목",
               "Summarize": "요약"
               }
    df_cls = df_cls.rename(columns=colDict)
    return df_cls

def merge_with_html_template(contents):
    """
    HTML 뉴스레터 템플릿 적용
    """
    f=codecs.open("./template/newsletter.html", 'r', 'utf-8')
    html = f.read().format(contents=contents)
    return html

def make_final_contents(search_word_list):
    
    # 컨텐츠 생성
    df = make_contents(search_word_list)
    # 전처리
    df_cls = preprocessing(df)
    # HTML로 변환하기
    html = df_to_html_table(df_cls)
    # HTML Contents
    contents_ = html.replace("&lt;","<").replace("&gt;",">")
    # 뉴스레터 HTML 템플릿 적용
    contents = merge_with_html_template(contents_)
    return contents