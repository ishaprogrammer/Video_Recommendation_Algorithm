# Video Recommendation System

## Overview
### This document provides a detailed explanation of the design, implementation, and key decisions made during the development of the video recommendation system. The system recommends motivational videos to users based on their preferences, interactions, and moods using a hybrid recommendation approach that combines content-based and collaborative filtering.

## Approach:

### Data Cleaning
I.	Data Loading & Inspection: Loaded datasets from API and CSV files, checked for missing values, duplicates, and irrelevant columns. Previewed data and standardized formats (e.g., last_login to datetime).
II.	Column Filtering & Missing Values: Dropped unnecessary columns (e.g., location, social links, identifiers) and filled missing values (e.g., forward-filled last login, set default values for interactions like False and 0).
III.	Data Merging: Consolidated user data, post metadata, and interaction datasets into a single cleaned DataFrame by merging on common keys, ensuring all relevant data was integrated for recommendation modeling.


### Data Preparation
-	The data used for recommendations is stored in a CSV file (`final_merged_df.csv`). This file contains:
-	User interaction data (e.g., username, view counts).
-	Video metadata (e.g., category, ratings, view counts).
-	The data is loaded into a Pandas DataFrame for processing.

### Recommendation Techniques
a. Content-Based Recommendations
-	Recommends videos based on user preferences, specifically their interaction history and favorite categories.
-	Algorithm:
1.	Identify categories frequently interacted with by the user.
2.	Select top-rated videos in those categories that the user has not yet viewed.
3.	Limit results to ensure diversity and relevance.

### Collaborative Filtering
-	Suggests videos based on interactions from users with similar preferences.
-	Algorithm:
1.	Create a user-item interaction matrix with ratings.
2.	Compute cosine similarity between users.
3.	Identify the most similar users and recommend videos they have liked but the current user has not viewed.
 
### Mood-Based Recommendations
-	Maps moods to predefined categories (e.g., "happy" maps to "Entertainment" and "Comedy").
-	Suggests popular videos in categories aligned with the user’s mood.

### Hybrid Recommendations
-	Combines content-based, collaborative, and mood-based recommendations.
-	Ensures diverse and relevant recommendations by:
-	Including top results from each method.
-	Filtering duplicate recommendations.
-	Ensuring a total of 10 recommendations.

### Evaluation Metrics
To assess recommendation quality:
-	Mean Absolute Error (MAE): Measures average error between predicted and actual ratings.
-	Root Mean Square Error (RMSE): Measures the square root of the average squared errors.
-	Both metrics are calculated on a test set split from the data.

### Model Architecture
-	The system is implemented using the FastAPI framework to serve API endpoints.
-	Key components:
-	Data Loader: Loads and validates the CSV file.
-	Metrics Class: Tracks predictions and calculates MAE and RMSE.
-	Recommendation Functions: Implements each recommendation technique.
-	API Endpoint: `/feed` generates recommendations and returns evaluation metrics.

### API Workflow
1.	Request Parameters:
-	`username`: Identifies the user requesting recommendations.
-	`category_id` (optional): Filters recommendations by category.
-	`mood` (optional): Suggests videos based on mood.
2.	Processing:
-	Loads user preferences and video data.
-	Computes recommendations using hybrid logic.
-	Filters results to ensure a diverse and relevant set of 10 videos.
3.	Response
-	Returns a JSON object with:
-	A list of 10 recommended videos.
-	Evaluation metrics (MAE, RMSE).

## Key Design Decisions:

1.	Hybrid Approach:
 
-	Combines multiple techniques to mitigate limitations of individual methods (e.g., cold start problems in collaborative filtering).
2.	Mood Integration:
-	Adds a personalized touch by incorporating mood-based recommendations.
3.	Evaluation Metrics
-	Includes both MAE and RMSE for a robust assessment of recommendation quality.

Challenges and Solutions
1.	Cold Start Problem:
-	Addressed by incorporating mood-based and content-based recommendations.
2.	Data Quality:
-	Ensured through validation during data loading.
3.	Scalability:
-	Used cosine similarity for efficient collaborative filtering computations.

Future Enhancements:
1.	Real-Time User Feedback:
-	Incorporate live feedback to refine recommendations dynamically.
2.	Advanced Models:
-	Leverage deep learning techniques for more accurate collaborative filtering.
3.	Expanded Mood Categories:
-	Enhance mood mappings for finer-grained recommendations.

## Conclusion
### The video recommendation system combines content-based, collaborative, and mood-based methods to deliver personalized suggestions. The hybrid approach ensures robustness, diversity, and relevance, making it a versatile solution for motivational content discovery.
