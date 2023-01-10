import pandas as pd
import ast

movie=pd.read_csv('insert movie dataframe address')
book=pd.read_csv('insert book dataframe address')


def pivoting(df):
    pivot_table=pd.DataFrame(columns=['제목'])
    for i in range(df.shape[0]):
        new=pd.DataFrame([(df['제목'][i])],columns=['제목'])
        pivot_table=pivot_table.append(new, ignore_index=True)
        review=ast.literal_eval(df['리뷰'][i])
        for j in range(len(review[0])):
            name=review[0][j][0]
            rating=review[0][j][1]
            if name in pivot_table.columns.to_list():
                pivot_table[name][i]=rating
            else:
                pivot_table[name]=0
                pivot_table[name][i]=rating
            
    return pivot_table

user_book_db=pivoting(book)
user_movie_db=pivoting(movie)
df=pd.concat([user_book_db, user_movie_db],axis=0)
user_book_db.to_csv('user_book address')
user_movie_db.to_csv('user_movie address')
df.to_csv('Final Dataframe address')
