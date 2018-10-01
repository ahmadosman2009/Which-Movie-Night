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

tmdb.API_KEY = 'd21af61e23e9064bcd06d07d27f70279'
#This is the function that outputs recommendations for my app. You'll replace this with your function that takes in the user input and gives the output
#def cos_sim_recommendations(new_data, df, index_name, n=1):
#    cs = cosine_similarity(new_data, df)
#    rec_index = np.argsort(cs)[0][-n-1:][::-1][1:]
#    recommendations = []
#    for rec in rec_index:
#        recommendations.append(index_name[rec])
#    return recommendations

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
#    result = dict(titles.iloc[movie_indices])
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
#     print(C)
#     print(m)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating2, axis=1)
#     print(qualified)
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

def filterGenre(dataframe, filterByGenre):
    dataframe['genres_x'] = dataframe['genres_x'].str.split('|')
    dataframe.reset_index(drop = True, inplace = True)
    row_list = []
    for i in dataframe.index:
        for genre in dataframe['genres_x'][i]:
            if genre == filterByGenre:
                row_list.append(i)
    df_new2 = dataframe.iloc[row_list]
#     df_new2 = df_new2.sample(n=20)
    df_new2.reset_index(drop = True, inplace = True)
#     df_new2.reset_index(drop = True, inplace = True)
    df_new2 = df_new2.sort_values('wr', ascending=False)
    df_new2 = df_new2[0:200]
    return df_new2.sample(n=30)


def topMovies(dataframe):
    dataframe['genres_x'] = dataframe['genres_x'].str.split('|')
    df_new2 = dataframe.sort_values('wr', ascending=False)
    df_new2.reset_index(drop = True, inplace = True)
    df_new2 = df_new2[0:150]
    return df_new2.sample(n=30)



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
#         print(full_df)
    return full_df

#Initialize app
#app = Flask(__name__, static_url_path='/static')
app = Flask(__name__)


#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
#    if request.method == 'POST':
#        numOfusers = request.form['numOfusers']
#        print(numOfusers)
#        return render_template('movieReview.html', numOfusers = numOfusers)
#    if request.method == 'POST':
#        age = request.form['age']
#        return render_template('movieReview.html', age = age)
    
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
    movie = tmdb.Movies(5)
    random_title = df_movies['title'][random_movie]
    
#    random_title = random_title.values
#    random_title = random_title.tolist()
#    
#    
#
    
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
    
    response = movie.info()
    movielink = movie.poster_path
    base_url = 'https://image.tmdb.org/t/p/w500'
#    total_link = base_url + movielink
    
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
    ratings = ['blah',
                'blah',
                'blah',
                'blah',
                'blah',
                'blah',
                'blah',
                'blah',
                'blah',
                'blah']
    
    
    results_from_java = request.get_json()
#    results_from_java = json["results"]
    
#    if request.method == 'POST':
#    age = request.form['age']
#    return render_template('movieReview.html', age = age)
    
    ##Filtering Data with selected genres from homepage:
    
#    'Children', 'Documentary', 'Animation', 'Mystery', '(no genres listed)', 
#    'Adventure', 'Thriller', 'Fantasy', 'Horror', 'Crime', 'Western', 'Film-Noir', 
#    'Sci-Fi', 'IMAX', 'Romance', 'War', 'Musical', 'Action', 'Drama', 'Comedy'
    genre_available = ['Action', 'Drama', 'Comedy','Romance','Thriller', 'Horror']
    
    genre_selected = []
    for i in genre_available:
#        print(i)
        if request.form.get(i):
#            action = i
            genre_selected.append(i)
        else:
#            action = 'None'
            genre_selected.append('None')
            
    for w in genre_selected: 
        if 'None' in genre_selected: 
            genre_selected.remove('None')
            
#    df_movies_posters = pd.read_csv('df_movies_posters_one_10k_new.csv')
    df_movies_posters = pd.read_csv('df_poster_toReview.csv')

    filtered_movies = topMovies(df_movies_posters)
#    filtered_movies = filterGenre(df_movies_posters, 'Comedy')
    selectionOfposter = filtered_movies['poster'].values.tolist()
#    selectionOftitles = filtered_movies['title'].values.tolist()
    selectionOftitles = filtered_movies['title_x'].values.tolist()

#    comdey= False
#    action= False
#    if request.form.get('Comdey'):
#        comdey = 'Comdey'
#        genre_selected.append('Comdey')
#    else:
#        comdey = 'None'
#        genre_selected.append('None')
        

        
    #this line below does not work if the user does not check it .. html fails
#    Action = request.form['Action']
    
#    movie1= Movie ('Toy Story (1995)')
    movie1= ["Toy Story","toy.jpg","1"]  
#    movie1 = json.dumps(movie1)
#    movie1= ['movie1','movie2']

    test =[1,2,3]
#    test = {'field1': 1, 'field2': 100}
#    test = json.dumps(test)
#        return render_template('movieReview.html', age = age, test = test, numOfusers =numOfusers, Action=Action, Comdey=Comdey, Drama=Drama)

    return render_template('movieReview.html',  results_from_java=results_from_java, 
                           random_title=movie_posters,total_link =total_link,
                           ratings=ratings,movie = movie1, age = age, test = test, 
                           numOfusers =numOfusers, genre_selected=genre_selected,
                           selectionOfposter=selectionOfposter,
                           selectionOftitles=selectionOftitles)
#
#After they submit the survey, the recommender page redirects to recommendations.html
@app.route('/movieRecommendations', methods=['GET', 'POST'])
def recommendations():
    my_list=[0,1,2,3,4,5]

#    vendors = pd.read_csv('data/florists_updated.csv')
#
#    # These are a couple examples of what the user input looks like.
#
#    #1. user enters the total price of the wedding. The name of the variable is arbitrary and doesn't have to be the same but is less confusing later on
#    numViewers = int(request.form['numViewers'])
#    total_price = float(request.form['total_price'])

#    #2. user also enters size of wedding
#    size_of_wedding = int(request.form['size_of_wedding'])
#
#    ### other user inputs
#    new_df = pd.DataFrame({
#                        'numberOfmovies':numberOfmovies})
##                        'size_of_wedding':size_of_wedding,
                        #etc....
                        
#    #call the recommender function
#    cos_sims = cos_sim_recommendations(new_df, X, index_name, n=1)
                        
#    numOfusers = int(request.form['numOfusers'])
                        
                        
                        
    age = request.form['age']
    movieToBaseRecOn = cleanMoviesUped(age)
#    for xx in range(len(movieToBaseRecOn)):

    title = 'Toy Story (1995)'
#    title = 'Ace Ventura: When Nature Calls (1995)'
#    recommend = improved_recommendations(title)
#    recommend = improved_recommendations(movieToBaseRecOn[1])
    recommend = genre_recommendations(title)
    recommend_one = recommend.values
    recommend_one = recommend_one.tolist()
    
#    ww = ['Toy Story (1995)','Ace Ventura: When Nature Calls (1995)']   
#    recommend=[]
#    for g in range(len(ww)):
#        output = improved_recommendations(ww[g])
#        recommend.extend(output)
#    recommend_one = recommend
    
##    df_movies_POSTERS = pd.read_csv('df_movies_posters_one_10k_new.csv')
    df_movies_POSTERS = pd.read_csv('posters_18k.csv')
    presentThese = movieToPoster(df_movies_POSTERS,recommend)
# #   presentThese = presentThese['poster'].values.tolist()
    presentThese = presentThese['poster_y'].values.tolist()
    
    
#    presentThese
#    #for my app the recommender gives an image of the florist and a link
#    florist_info = {
#    'Flora_by_Nora':
#        {'name':'Flora by Nora', 'img_src':'/static/img/flora_by_nora.png', 'link':'https://www.florabynora.com/'},
#    'Madelyn_Claire_Floral_Design_&_Events':
#        {'name':'Madlyn Claire Floral Design', 'img_src':'/static/img/madelyn_claire.png', 'link':'https://madelynclairefloraldesign.com/'}}
#
#
#    #arguments are whatever comes out of your app, in my case a cos_sim and the recommended florist
#    #the structure is render_template('your_html_file.html', x=x, y=y, etc...)
#    #refer to my recommendations.html to see how variables work
#    return render_template('movieRecommendations.html', cos_sims = cos_sims, florist_info = florist_info)
    return render_template('movieRecommendations.html',presentThese=presentThese,
                           age=age, recommend_pass = recommend_one)
    

#@app.route('/movieRecommendations', methods=['GET', 'POST'])
#def recommendations():
#    df_movies = pd.read_csv('movies_26M.csv')
#    title = int(request.form['title'])
#    recommend = genre_recommendations(title)
#    return render_template('movieRecommendations.html', recommend =recommend)



if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8855, debug=True)
    
#    http://0.0.0.0:8080/
