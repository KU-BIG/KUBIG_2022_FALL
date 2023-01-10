from movie import movie
import pandas as pd


url_list=movie.getting_movieurl('https://pedia.watcha.com/ko-KR/users/DgwxAeQYNxrMj/contents/movies/ratings')
overview=movie.movie_overview(url_list)
overview.to_csv('insert your local address')
