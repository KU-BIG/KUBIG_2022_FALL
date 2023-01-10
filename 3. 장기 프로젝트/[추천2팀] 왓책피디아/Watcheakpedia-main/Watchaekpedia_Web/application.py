from flask import Flask, render_template,url_for
import pandas as pd
import numpy as np
import sys
import ast


application = Flask(__name__)


@application.route("/")
def main():
    df=pd.read_csv('final.csv')
    item=pd.read_csv('item.csv')
    users=df.columns
    i=np.random.randint(1,8629)
    user=users[i]
    recommend=ast.literal_eval(df[users[i]][0])
    content=[];link=[];idxes=[]
    for book in recommend:
        idx=item[(item['제목']==book)].index.values
        content.append(item['내용'][idx].values[0])
        link.append(item['링크'][idx].values[0])
        idxes.append(idx[0])
        
        
    
    return render_template("index.html",user=user, link1=link[0], idx1="static/img/"+str(idxes[0])+".jpg",book1=recommend[0],
                          book1_information=content[0], link2=link[1],idx2="static/img/"+str(idxes[1])+".jpg",
                          book2=recommend[1], book2_information=content[1], link3=link[2], idx3="static/img/"+str(idxes[2])+".jpg",
                          book3=recommend[2], book3_information=content[2], link4=link[3], idx4="static/img/"+str(idxes[3])+".jpg",
                          book4=recommend[3], book4_information=content[3])


if __name__ == "__main__":
    application.run(host='0.0.0.0', debug=True)
