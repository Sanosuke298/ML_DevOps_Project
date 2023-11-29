# Steam Analytics and Recommendation System

[![Data Science](https://img.shields.io/badge/Data%20Science-Advanced-brightgreen?style=for-the-badge&logo=datascience)](https://www.example.com)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Expert-blue?style=for-the-badge&logo=python)](https://www.example.com)
[![Natural Language Processing](https://img.shields.io/badge/NLP-Proficient-yellow?style=for-the-badge&logo=naturallanguageprocessing)](https://www.example.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Integration-orange?style=for-the-badge&logo=fastapi)](https://www.example.com)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue?style=for-the-badge&logo=docker)](https://www.example.com)
[![Render](https://img.shields.io/badge/Render-Deployment-success?style=for-the-badge&logo=render)](https://www.example.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Manipulation-yellowgreen?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Operations-blue?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/stable/)
[![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-yellow?style=for-the-badge&logo=nltk)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP%20Library-green?style=for-the-badge&logo=spacy)](https://spacy.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-lightgrey?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)
[![Parquet](https://img.shields.io/badge/Parquet-Data%20Storage%20Format-ff69b4?style=for-the-badge&logo=apache)](https://parquet.apache.org/)

## Table of Contents

- [Project Overview](#project-overview)
  - [Project Goals](#project-goals)
  - [Key Features](#key-features)
  - [Technological Stack](#technological-stack)
  - [Project Workflow](#project-workflow)
  - [Impact](#impact)
- [Data Sources](#data-sources)
- [Extraction, Transformation, and Loading (ETL)](#extraction-transformation-and-loading-etl)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Function Creation](#function-creation)
- [Machine Learning Modeling](#machine-learning-modeling)
  - [Game Recommendation](#game-recommendation)
  - [User Recommendation](#user-recommendation)
- [Deployment](#deployment)
- [Video](#video)
- [Conclusions](#conclusions)
- [Links](#links)

# Project Overview

In an era where digital entertainment choices are abundant yet fragmented, the **Steam Game Recommendation System** stands as a pioneering effort to harness data science and machine learning for enhanced user experience in digital game selection. This project is not just a testament to technical prowess in handling complex datasets, but also a reflection of the nuanced understanding of user behavior and preferences.

## Project Goals

- **Sentiment Analysis**: Utilizing Natural Language Processing (NLP) techniques, the project deciphers user sentiments from comments and reviews. This sentiment analysis provides valuable insights into user feedback, enabling data-driven decision-making.
- **Game Recommendation System**: At its core, this project aims to create a game recommendation system tailored for the Steam platform. Leveraging the power of Machine Learning, this system offers personalized game suggestions, elevating user engagement and satisfaction.

## Key Features

- **MLOps Engineering**: This project simulates the role of an MLOps Engineer, seamlessly blending responsibilities of a Data Engineer and Data Scientist within the Steam gaming platform environment.
- **Minimum Viable Product (MVP)**: The primary objective is to develop a Minimum Viable Product (MVP) comprising an implemented API and a Machine Learning model. This MVP focuses on sentiment analysis based on user comments and provides a game recommendation system for the Steam platform.

## Technological Stack

- **Data Science Frameworks**: Python, Jupyter Notebook
- **Machine Learning Libraries**: scikit-learn, TensorFlow
- **Natural Language Processing (NLP)**: NLTK, spaCy
- **Web Framework**: FastAPI
- **Containerization**: Docker
- **Deployment**: Render
- **Data Manipulation and Analysis**: pandas, NumPy
- **Text Processing**: TfidfVectorizer
- **Data Visualization**: Matplotlib, Seaborn
- **Data Storage**: Parquet, SQLite

# Project Workflow

1. **Data Aggregation**: Gathering gaming data from various sources within the Steam platform.
2. **Data Cleansing and Preprocessing**: Cleaning and transforming raw data to ensure it's suitable for analysis.
3. **Exploratory Data Analysis (EDA)**: Gaining insights from data through visualization and statistical analysis.
4. **User Review Sentiment Analysis**: Applying NLP techniques to understand user sentiments from comments and reviews.
5. **Machine Learning Model**: Implementing a Machine Learning model, including the use of Cosine Similarity, to recommend games based on user preferences.
6. **API Development**: Creating an API to provide real-time game recommendations to users.
7. **User-Centric Insights**: Focusing on user behavior analysis, user engagement metrics, and user-centric data processing.
8. **Data-Driven Decision-Making**: Empowering decision-making through data-driven insights and personalized gaming experiences.

## Data Sources

In this initial stage, we define the data sources that form the foundation of our project. These sources consist of three JSON files, each containing distinct yet interconnected information:

- **output_steam_games.json**: This file serves as a comprehensive database of game-related details, encompassing information such as game titles, developers, genres, and tags. It forms the backbone of our game recommendation system.
- **australian_users_items.json**: Here, we find data related to user interactions with games, including playtime and ownership. This dataset allows us to understand user behavior and preferences.
- **australian_users_reviews.json**: This dataset contains user-generated reviews and recommendations, offering insights into user sentiment and feedback.

### Extraction, Transformation, and Loading (ETL)

The ETL phase is the fundamental process of extracting, transforming, and loading data from its raw state to a format suitable for analysis. In this phase:

- **Data Extraction**: We gather data from the initial JSON files, preparing it for subsequent analysis. This step ensures we have access to the necessary data for our project.
- **Data Transformation**: Data is cleaned and structured to eliminate inconsistencies and ensure readability. In particular, the 'NLTK' library is used for sentiment analysis on user comments, resulting in the creation of a 'sentiment_analysis' column.
- **Data Loading**: Cleaned and transformed datasets are stored in 'parquet' format, optimizing storage and retrieval efficiency.

### Exploratory Data Analysis (EDA)

EDA is a crucial step in understanding the characteristics and insights hidden within our data. During this phase:

- **Data Analysis**: We explore the datasets to gain a deeper understanding of the variables they contain. This includes visualizations, statistical analysis, and identifying patterns or trends.
- **Feature Engineering**: Essential variables are identified for use in subsequent phases, particularly for building the recommendation model.

### Function Creation

In this stage, we create specific functions tailored to our project's objectives. These functions enable us to perform targeted analyses and provide meaningful insights to users:

- **PlayTimeGenre**: Calculates and returns the year with the highest playtime for a given genre. This helps users identify popular years for specific game genres.
- **UserForGenre**: Determines the user with the most playtime for a particular genre, offering insights into genre-specific user engagement over the years.
- **UserRecommend**: Provides game recommendations for a specific year, enhancing user game discovery.
- **UsersWorstDeveloper**: Identifies the top 3 worst game developers based on sentiment analysis, helping users avoid low-rated developers.
- **sentiment_analysis**: Conducts sentiment analysis on user reviews, categorizing them as negative, neutral, or positive. This offers insights into user sentiment trends over time.

## Machine Learning Modeling

### Game Recommendation

The `game_recommendation` function is a critical component of our project, designed to enhance the gaming experience for users. Its primary objective is to recommend games to users based on their preferences and the games they have interacted with. Here's how it works:

- **Input**: Users provide an item ID, representing a game they have played or are interested in.
- **Functionality**:

  - The function takes the provided item ID and checks if it exists in our dataset. If not, it raises a 404 error, indicating that the game is not found in our data.
  - Using cosine similarity, the function identifies games similar to the provided one, ensuring a tailored recommendation.
  - The top 5 recommended games are selected based on their similarity scores, excluding the initially provided game to ensure diversity.
  - The recommendations, along with a personalized message, are returned to the user.
- **Output**: Users receive a list of game recommendations with a message that says, "If you liked the game [provided game], you might also like:" followed by the recommended games. This feature facilitates game discovery and aligns with our project's goal of enhancing the gaming experience.

## User Recommendation

The `user_recommendation` function caters to the specific needs of individual users, providing them with personalized game recommendations. Here's how it operates:

- **Input**: Users provide their user ID, allowing the system to generate recommendations tailored to their gaming history.
- **Functionality**:

  - The function begins by checking if the user ID is present in our dataset. If not, it raises a 404 error, indicating that the user is not found in our data.
  - The function then identifies games played or interacted with by the user.
  - If the user has not interacted with enough games (less than 5), the function returns a message indicating that there is insufficient data to generate recommendations.
  - To generate recommendations, the function creates a user profile vector. This vector is constructed by averaging the TF-IDF vectors of the games the user has interacted with.
  - Cosine similarity scores are calculated between the user's profile vector and all games in our dataset.
  - The games are sorted based on similarity scores, and the top 5 unique games (excluding those the user has already interacted with) are selected as recommendations.
- **Output**: Users receive a list of the top 5 recommended games, personalized to their gaming history. This feature aims to enhance user engagement and satisfaction on the Steam gaming platform.

These functions add valuable user-centric features to our project, focusing on both game-specific and user-specific recommendations.

## Deployment

Finally, the project is deployed on the Render platform via Docker Hub. This deployment ensures that the game recommendation system is accessible and usable by users, offering a seamless and interactive experience.

Each stage in the data pipeline plays a vital role in achieving the project's goals, from data preparation and exploration to user-centric functionality and machine learning-powered recommendations.

Link to the API: [API Documentation](https://my-new-app-jlt5.onrender.com/docs)

## Video

A video explaining and demonstrating the API is available. 
[![Video Preview](https://img.youtube.com/vi/ANScEWm5W3s/0.jpg)](https://www.youtube.com/watch?v=ANScEWm5W3s)


## Conclusions

This project applies knowledge from the HENRY Data Science program, covering typical Data Engineer and Data Scientist responsibilities. The successful creation of an MVP including an API and a web service implementation is a significant achievement, though there's room for further optimization and efficiency improvements.
