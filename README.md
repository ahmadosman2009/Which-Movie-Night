# Which-Movie-Night
## Movie Recommendation System that combines movie features liked by multiple users (2-6) and suggest movies to watch.

Used 45k movies from MovieLens database, and APIs of IMDB and The Movie DB.

Web application user interface using Flask/HTML/CSS/JavaScript and hosted on AWS.

Models Tested:
1) Content based recommedations.
2) Memory based collaborative filtering (user based)    RMSE:654.9
3) Memory based collaborative filtering (movie based)   RMSE:114.7
4) Model based collaborative filtering using SVD        RMSE: 0.95
5) Deep learning.                                       RMSE: 3.68
