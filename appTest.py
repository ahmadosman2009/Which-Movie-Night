from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler

import tmdbsimple as tmdb

pd.options.display.max_columns=25


def genre_recommendations(title):
    df_movies = pd.read_csv('movies_26M.csv')
    titles = df_movies['title']
    indices = pd.Series(df_movies.index, index=df_movies['title'])
    cosine_sim = pd.read_csv('cosine_sim_small.csv')
    
    idx = indices[title]
    cosine_sim = cosine_sim.values
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def weighted_rating2(x):
    v = x['vote_count']
    R = x['vote_average']
    m=50
    C=2.5
    return (v/(v+m) * R) + (m/(m+v) * C)
    

def improved_recommendations(title):
    new_df3 = pd.read_csv('df_Soup_18K.csv')
    cosine_sim = pd.read_csv('cosine_sim_Soup_18K.csv')
    cosine_sim = cosine_sim.values 
    
    titles = new_df3['title_x']
    indices = pd.Series(new_df3.index, index=new_df3['title_x'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = new_df3.iloc[movie_indices][['title_x', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
#     m = vote_counts.quantile(0.60)
    m = 50
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating2, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified['title_x']



#def filterGenre(dataframe, filterByGenre):
#    dataframe['genres'] = dataframe['genres'].str.split('|')
#    dataframe.reset_index(drop = True, inplace = True)
#    row_list = []
#    for i in dataframe.index:
#        for genre in dataframe['genres'][i]:
#            if genre == filterByGenre:
#                row_list.append(i)
#    df_new = dataframe.iloc[row_list]
#    df_new = df_new.sample(n=20)
#    df_new.reset_index(drop = True, inplace = True)
#    return df_new
#
#def filterGenre(dataframe, filterByGenre):
#    dataframe['genres_x'] = dataframe['genres_x'].str.split('|')
#    dataframe.reset_index(drop = True, inplace = True)
#    row_list = []
#    for i in dataframe.index:
#        for genre in dataframe['genres_x'][i]:
#            if genre == filterByGenre:
#                row_list.append(i)
#    df_new2 = dataframe.iloc[row_list]
##     df_new2 = df_new2.sample(n=20)
#    df_new2.reset_index(drop = True, inplace = True)
##     df_new2.reset_index(drop = True, inplace = True)
#    df_new2 = df_new2.sort_values('wr', ascending=False)
#    df_new2 = df_new2[0:200]
#    return df_new2.sample(n=30)

def filterGenre2(dataframe, filterByGenre):
    dataframe.reset_index(drop = True, inplace = True)
    row_list = []
    for i in dataframe.index:
        for genre in dataframe['genres'][i]:
            if genre == filterByGenre:
                row_list.append(i)
    df_new = dataframe.iloc[row_list]
    df_new = df_new.sort_values('wr', ascending=False)
    df_new = df_new[0:70].sample(n=30)
#     df_new.reset_index(drop = True, inplace = True)
    return df_new

#def topMovies(dataframe):
#    dataframe['genres_x'] = dataframe['genres_x'].str.split('|')
#    df_new2 = dataframe.sort_values('wr', ascending=False)
#    df_new2.reset_index(drop = True, inplace = True)
#    df_new2 = df_new2[0:150]
#    return df_new2.sample(n=30)


def topMovies2(dataframe):
#     dataframe['genres'] = dataframe['genres'].str.split('|')
    dataframe2 = dataframe.sort_values('wr', ascending=False)
    dataframe2.reset_index(drop = True, inplace = True)
    return dataframe2[0:150].sample(n=35)

def cleanMoviesUped(movieString):
    movieStringnew = movieString.split('),')
    for c in range(len(movieStringnew)-1):
        movieStringnew[c] = movieStringnew[c]+')'
    return movieStringnew


#takes in output from recommendation function and outputs posters 
def movieToPoster(movieDf,recommenderOutput):
    recommenderOutput = pd.DataFrame(recommenderOutput)
    recommenderOutput.reset_index(drop = True, inplace = True)
    full_df=pd.DataFrame()
    for z in range(len(recommenderOutput)):
#         print(z)
        new_df = movieDf.loc[movieDf['title'] == recommenderOutput['title'][z]]
#        new_df = movieDf.loc[movieDf['title'] == recommenderOutput['title_x'][z]]
        full_df = full_df.append(new_df)
        full_df.reset_index(drop = True, inplace = True)
        full_df = full_df.sort_values('wr', ascending=False)
#         print(full_df)
    return full_df

#Initialize app
app = Flask(__name__)


#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
    
    return render_template('homepage.html')
#
#My home page redirects to recommender.html where the user fills out a survey (user input)
@app.route('/movieReview', methods=['GET', 'POST'])
def movieReview():  
#    test = ['static/img/Mom.jpg','static/img/PulpFiction.jpg','static/img/SavingPrivateRyan.jpg']
    test =['dog','cat','mouse']
    numOfusers = request.form['numOfusers']
    age = request.form['age']
    
#    dude = request.form['dude']
    df_movies = pd.read_csv('movies_26M.csv')
#    random_movie = df_movies['movieId'].sample(10)
    random_movie = 1
#    movie = tmdb.Movies(5)
    random_title = df_movies['title'][random_movie]
    
    
    movie_posters = ['static/img/Mohana.jpg',
                     'static/img/HarryPotter.jpg',
                     'static/img/SavingPrivateRyan.jpg',
                     'static/img/10.jpg',
                     'static/img/11.jpg',
                     'static/img/12.jpg',
                     'static/img/13.jpg',
                     'static/img/14.jpg',
                     'static/img/15.jpg',
                     'static/img/16.jpg']
    
    
    total_link = ['Mohana',
                  'HarryPotter',
                  'SavingPrivateRyan',
                  'movie10',
                  'movie11',
                  'movie12',
                  'movie13',
                  'movie14',
                  'movie15',
                  'movie16']
    
    
    results_from_java = request.get_json()
#    results_from_java = json["results"]
    
    
    ##Filtering Data with selected genres from homepage:
#    'Children', 'Documentary', 'Animation', 'Mystery', '(no genres listed)', 
#    'Adventure', 'Thriller', 'Fantasy', 'Horror', 'Crime', 'Western', 'Film-Noir', 
#    'Sci-Fi', 'IMAX', 'Romance', 'War', 'Musical', 'Action', 'Drama', 'Comedy'
    genre_available = ['Action', 'Drama', 'Comedy','Romance','Thriller', 'Horror']
    
    genre_selected = []
    final_genre = []
    for i in genre_available:
        if request.form.get(i):
            genre_selected.append(i)
        else:
            genre_selected.append('None')
            
    for w in range(len(genre_selected)): 
        if genre_selected[w] != 'None':
            final_genre.append(genre_selected[w])
        
            
            
    df_movies_posters = pd.read_csv('posters_18k_clean2.csv')
    df_movies_posters['genres'] = df_movies_posters['genres'].str.split('|')
    list_of_df = []
    for y in final_genre:
        output = filterGenre2(df_movies_posters, y)
        list_of_df.append(output)
    all_filtered_movies = pd.concat(list_of_df) 
    all_filtered_movies_order = all_filtered_movies.sort_values('wr', ascending=False)
    filtered_movies = all_filtered_movies_order.sample(n=30)
    
    
#    filtered_movies = topMovies2(df_movies_posters)
#    filtered_movies = filterGenre(df_movies_posters, 'Comedy')
#    filtered_movies = filterGenre2(df_movies_posters, 'Comedy')
    selectionOfposter = filtered_movies['poster_y'].values.tolist()
#    selectionOfposter = filtered_movies['poster'].values.tolist()
    selectionOftitles = filtered_movies['title'].values.tolist()
#    selectionOftitles = filtered_movies['title_x'].values.tolist()

#    comdey= False
#    action= False
#    if request.form.get('Comdey'):
#        comdey = 'Comdey'
#        genre_selected.append('Comdey')
#    else:
#        comdey = 'None'
#        genre_selected.append('None')
        

    
    movie1= ["Toy Story","toy.jpg","1"]  


    test =[1,2,3]

    return render_template('movieReview.html',  results_from_java=results_from_java, 
                           random_title=movie_posters,total_link =total_link,
                           movie = movie1, age = age, test = test, 
                           numOfusers =numOfusers, genre_selected=genre_selected,
                           selectionOfposter=selectionOfposter,
                           selectionOftitles=selectionOftitles,final_genre=final_genre)
#
#After they submit the survey, the recommender page redirects to recommendations.html
@app.route('/movieRecommendations', methods=['GET', 'POST'])
def recommendations():
    my_list=[0,1,2,3,4,5]

                        
#    numOfusers = int(request.form['numOfusers'])
                        
                        
                        
    age = request.form['age']
    movieToBaseRecOn = cleanMoviesUped(age)
#    for xx in range(len(movieToBaseRecOn)):
    
#    list_df = []
#    for k in movieToBaseRecOn:
#        output = improved_recommendations(k)
#        list_df.append(output)
#    all_newoutput = pd.concat(list_df)
## #   all_newoutput_order = all_newoutput.sort_values('wr', ascending=False)
#    recommend = all_newoutput.sample(n=16)

    title = 'Toy Story (1995)'
#    title = 'Ace Ventura: When Nature Calls (1995)'
#    recommend = improved_recommendations(title)
#    recommend = improved_recommendations(movieToBaseRecOn[1])
    recommend = genre_recommendations(title)
    recommend_one = recommend.values
    recommend_one = recommend_one.tolist()
#    movieToBaseRecOn = ['Scarface (1983)', 'Pulp Fiction (1994)', 'Monsters, Inc. (2001)']

##    df_movies_POSTERS = pd.read_csv('df_movies_posters_one_10k_new.csv')
    df_movies_POSTERS = pd.read_csv('posters_18k_clean2.csv')
    presentTheseOutput = movieToPoster(df_movies_POSTERS,recommend)
# #   presentThese = presentThese['poster'].values.tolist()
    presentThese = presentTheseOutput['poster_y'].values.tolist()
    presentTheseLinks = presentTheseOutput['link'].values.tolist()

    

    return render_template('movieRecommendations.html',presentThese=presentThese,
                           age=age, recommend_pass = recommend_one,
                           presentTheseLinks=presentTheseLinks)
    



if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8555, debug=True)
    