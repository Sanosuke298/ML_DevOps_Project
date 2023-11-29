from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "https://my-new-app-jlt5.onrender.com",  # Add your deployment URL here
    # Add other domains if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load parquet data into a pandas dataframe
games_data = pd.read_parquet('API_requests.parquet')

# Prepare data for recommendation
games_data_rec = games_data.copy()
games_data_rec.drop(['user_id', 'playtime_forever', 'recommend', 'review', 'year'], axis=1, inplace=True)
games_data_rec['item_id'] = games_data_rec['item_id'].astype(str)
games_data_rec['year_posted'] = games_data_rec['year_posted'].astype(str)
games_data_rec.drop_duplicates(subset='item_id', keep='first', inplace=True)
games_data_rec['features'] = games_data_rec[['title', 'year_posted', 'developer']].agg(', '.join, axis=1)
games_data_rec.drop(['title', 'year_posted', 'developer'], axis=1, inplace=True)

# Create TF-IDF and cosine similarity matrix
tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')
data_vector = tfidv.fit_transform(games_data_rec['features'])
data_vector_df = pd.DataFrame(data_vector.toarray(), index=games_data_rec['item_id'])
cos_sim_df = pd.DataFrame(cosine_similarity(data_vector_df), index=data_vector_df.index, columns=data_vector_df.index)

@app.get("/", summary="Root Endpoint")
def read_root():
    """ Returns a welcome message. """
    return {"message": "Welcome to my FastAPI application!"}

@app.get("/PlayTimeGenre", summary="Play Time by Genre")
def play_time_by_genre(genre: str):
    """ Returns the year with the most playtime for a given genre. """
    df_genre = games_data[games_data['genres'].str.contains(genre, case=False, na=False)]

    if df_genre.empty:
        raise HTTPException(status_code=404, detail="No data for the specified genre.")

    year_max_playtime = df_genre.groupby('year')['playtime_forever'].sum().idxmax()

    return {'Year with most play hours for Genre': genre, 'Year': int(year_max_playtime)}

@app.get("/UserForGenre", summary="User For Genre")
def user_for_genre(genre: str):
    """ Returns the user with the most play hours for a given genre. """
    df_genre = games_data[games_data['genres'].str.contains(genre, case=False, na=False)]

    if df_genre.empty:
        raise HTTPException(status_code=404, detail="No data for the specified genre.")

    user_max_hours = df_genre.groupby('user_id')['playtime_forever'].sum().idxmax()
    hours_per_year = df_genre.groupby('year_posted')['playtime_forever'].sum().reset_index().to_dict('records')

    return {
        'User with most play hours for Genre': genre,
        'User ID': user_max_hours,
        'Play Hours by Year': hours_per_year
    }

@app.get("/UsersRecommend", summary="Users Recommend")
def users_recommend(year: int):
    """ Returns the top 3 games with the most positive recommendations for a given year. """
    df_year = games_data[(games_data['year_posted'] == year) & (games_data['recommend'])]

    df_count = df_year.groupby('title')['recommend'].count()
    top_games = df_count.nlargest(3).index.tolist()

    return {'Top Recommended Games': top_games}

@app.get("/UsersWorstDeveloper", summary="Users Worst Developer")
def users_worst_developer(year: int):
    """ Returns the top 3 developers with the most negative recommendations for a given year. """
    df_year = games_data[(games_data['year_posted'] == year) & (~games_data['recommend'])]

    df_count = df_year.groupby('developer')['recommend'].count()
    worst_developers = df_count.nlargest(3).index.tolist()

    return {'Top 3 Developers with Negative Reviews': worst_developers}

@app.get("/sentiment_analysis", summary="Sentiment Analysis")
def sentiment_analysis(developer: str):
    """ Returns the sentiment analysis (positive, neutral, negative) for a given developer. """
    df_developer = games_data[games_data['developer'] == developer]

    if df_developer.empty:
        raise HTTPException(status_code=404, detail="No data for the specified developer.")

    sentiment_counts = df_developer['review'].value_counts().to_dict()

    return {
        'Developer': developer,
        'Sentiment Counts': {
            'Negative': sentiment_counts.get(0, 0),
            'Neutral': sentiment_counts.get(1, 0),
            'Positive': sentiment_counts.get(2, 0)
        }
    }

@app.get("/game_recommendation", summary="Game Recommendation")
def game_recommendation(item_id: int):
    """ Returns game recommendations based on a given item ID. """
    item_id_str = str(item_id)

    if item_id_str not in games_data_rec['item_id'].values:
        raise HTTPException(status_code=404, detail="Item ID not found in the data.")

    similar_games = cos_sim_df.loc[item_id_str].sort_values(ascending=False).head(6)
    result_df = similar_games.reset_index().merge(games_data_rec, on='item_id')

    game_title = result_df[result_df['item_id'] == item_id_str]['features'].values[0].split(', ')[0]
    recommendations = result_df['features'][1:6].apply(lambda x: x.split(', ')[0]).tolist()

    return {
        'message': f"If you liked the game {item_id_str} : {game_title}, you might also like:",
        'recommended games': recommendations
    }

@app.get("/user_recommendation", summary="User Recommendation")
def user_recommendation(user_id: int):
    """ Returns the top 5 recommended games for a given user ID. """
    if user_id not in games_data['user_id'].values:
        raise HTTPException(status_code=404, detail="User ID not found in the data.")

    # Filter games played or interacted with by the user
    user_games = games_data[games_data['user_id'] == user_id]

    # If the user has not interacted with enough games, handle this case
    if user_games.empty or len(user_games) < 5:
        return {"message": "Not enough data to generate recommendations for this user."}

    # Create a user profile vector
    # Here, we'll simply average the TF-IDF vectors of the games the user has interacted with
    user_profile_vector = data_vector_df.loc[user_games['item_id'].astype(str)].mean(axis=0)

    # Calculate similarity scores between the user profile and all games
    similarity_scores = cosine_similarity(user_profile_vector.values.reshape(1, -1), data_vector_df)

    # Sort the games based on similarity scores
    sorted_indices = np.argsort(similarity_scores.flatten())[::-1]

    # Select the top 5 unique games, excluding the ones the user has already interacted with
    recommended_indices = [idx for idx in sorted_indices if idx not in user_games['item_id'].values][:5]
    recommended_games = games_data_rec.iloc[recommended_indices]['features'].apply(lambda x: x.split(', ')[0]).tolist()

    # Return the recommended games
    return {
        'User ID': user_id,
        'Recommended Games': recommended_games
    }
