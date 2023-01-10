import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class movie:
 def __init__(self, address):
  self.address=address #driver address
 
 def getting_movieurl(url):
  url_list=[]
  driver = webdriver.Chrome(self.address)
  driver.get(url)
  import time
  scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
  while True:
      driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

      time.sleep(1.5)
      new_scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
      
      if scroll_pane_height == new_scroll_pane_height:
        break

      scroll_pane_height = new_scroll_pane_height
  for i in range(1,5434):
     try: 
      url='//*[@id="root"]/div/div[1]/section/section/div[1]/section/div[1]/div/ul/li['+str(i)+']/a' 
      find_href=driver.find_elements("xpath", url) 
      for my_href in find_href:
        url_js=my_href.get_attribute("href") #href 속성 추출
        url_list.append(url_js)
     except:
       None

  return url_list

 def movie_overview(book_page_urls):
  rows=[]
  for book_page_url in book_page_urls:
    try:
      html = urlopen(book_page_url)
      bsObject = BeautifulSoup(html, "html.parser")
      title = bsObject.find('h1', {'class':'css-171k8ad-Title e1svyhwg17'}).text
      html = urlopen(book_page_url+'/overview')
      bsObject = BeautifulSoup(html, "html.parser")
      dl=bsObject.find('ul',{'class':"css-f4q6l4-VisualUl-DescriptionUl e1kvv3950"}).find_all('dt')
      dd=bsObject.find('ul',{'class':"css-f4q6l4-VisualUl-DescriptionUl e1kvv3950"}).find_all('dd')
      original_title, year, country, genre, running_time, audience, content='','','','','','',''
      for ix in range(len(dl)):
        if dl[ix].text=='원제':
            original_title=dd[ix].text
        elif dl[ix].text=='장르':
            genre=dd[ix].text
        elif dl[ix].text=='국가':
            country=dd[ix].text
        elif dl[ix].text=='제작 연도':
            year=dd[ix].text
        elif dl[ix].text=='상영시간':
            running_time=dd[ix].text
        elif dl[ix].text=='연령 등급':
            audience=dd[ix].text
        elif dl[ix].text=='내용':
            content=dd[ix].text
      import time
      driver = webdriver.Chrome('/Users/jungjunlim/desktop/Python/[Recsys] Crawling/chromedriver')
      driver.get(book_page_url+'/comments')
      scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
      while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

        time.sleep(1)
        new_scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
      
        if scroll_pane_height == new_scroll_pane_height:
          break

        scroll_pane_height = new_scroll_pane_height

      html = driver.page_source
      soup = BeautifulSoup(html, 'lxml')
      id=soup.find_all('div', {'class':"css-1agoci2"}) #user_id
      ratings=soup.find_all('div', {'class':"css-yqs4xl"}) #별점
      reviews = soup.find_all("div", {"class":"css-1g78l7j"}) #리뷰내용
      minirows=[]
      minirows2=[]
      for i in range(len(id)):
       try:
        row=[id[i].get_text(),ratings[i].get_text(),reviews[i].get_text()]
        minirows.append(row)
       except:
         None
      minirows2.append(minirows)
      row = [title, original_title, year, country, genre, running_time, audience, content,minirows2]
      rows.append(row)
    except:
      None 
  df = pd.DataFrame(rows,columns=['제목','원제','제작연도','국가','장르','상영시간','연령등급','내용','리뷰'])
  return df

 def movie_ratings(book_page_urls):
  rows=[]
  for book_page_url in book_page_urls:
    #무한 스크롤러
    import time
    driver = webdriver.Chrome('/Users/jungjunlim/desktop/Python/[Recsys] Crawling/chromedriver')
    driver.get(book_page_url+'/comments')
    scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
    while True:
      driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

      time.sleep(1)
      new_scroll_pane_height = driver.execute_script('return document.body.scrollHeight')
      
      if scroll_pane_height == new_scroll_pane_height:
        break

      scroll_pane_height = new_scroll_pane_height

    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    id=soup.find_all('div', {'class':"css-1agoci2"}) #user_id
    ratings=soup.find_all('div', {'class':"css-yqs4xl"}) #별점
    reviews = soup.find_all("div", {"class":"css-1g78l7j"}) #리뷰내용
    minirows=[]
    minirows2=[]
    for i in range(len(id)):
     try:
      row=[id[i].get_text(),ratings[i].get_text(),reviews[i].get_text()]
      minirows.append(row)
     except:
       None
    minirows2.append(minirows)
    rows.append(minirows2)
    df=pd.DataFrame(rows,columns=['Review List'])
  return df

