import pandas as pd
from book import book


url_list=book.getting_movieurl('https://pedia.watcha.com/ko-KR/users/DgwxAeQYNxrMj/contents/books/ratings')
overview=book.book_overview(url_list)
overview.to_csv('insert your local address')

