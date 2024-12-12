from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Dict, Tuple
import uvicorn
from sklearn.model_selection import train_test_split
import os

app = FastAPI()

def load_data():
    file_path = "final_merged_df.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist in the current directory.")
    
    try:
        final_merged_df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return final_merged_df
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

class RecommendationMetrics:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        
    def calculate_mae(self) -> float:
        if not self.predictions or not self.actuals:
            return 0.0
        return np.mean(np.abs(np.array(self.predictions) - np.array(self.actuals)))
    
    def calculate_rmse(self) -> float:
        if not self.predictions or not self.actuals:
            return 0.0
        return np.sqrt(np.mean((np.array(self.predictions) - np.array(self.actuals)) ** 2))
    
    def add_prediction(self, predicted: float, actual: float):
        self.predictions.append(predicted)
        self.actuals.append(actual)
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'mae': self.calculate_mae(),
            'rmse': self.calculate_rmse()
        }

metrics = RecommendationMetrics()

def evaluate_recommendations(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    test_ratings = test_df['rating_percent'].values
    
    predictions = []
    for _, row in test_df.iterrows():
        similar_posts = train_df[train_df['category_id'] == row['category_id']]
        if not similar_posts.empty:
            pred_rating = similar_posts['rating_percent'].mean()
        else:
            pred_rating = train_df['rating_percent'].mean()
        predictions.append(pred_rating)
    
    for pred, actual in zip(predictions, test_ratings):
        metrics.add_prediction(pred, actual)
    
    return metrics.get_metrics()

def get_user_preferences(df: pd.DataFrame, username: str) -> pd.DataFrame:
    user_views = df[df['username'] == username]
    return user_views

def get_content_based_recommendations(df: pd.DataFrame, user_prefs: pd.DataFrame) -> List[int]:
    if user_prefs.empty:
        return df.nlargest(10, 'view_count')['post_id'].unique().tolist()
    
    user_categories = user_prefs['category_id'].value_counts().index.tolist()
    
    recommendations = []
    for category in user_categories:
        category_posts = df[
            (df['category_id'] == category) & 
            (~df['post_id'].isin(user_prefs['post_id']))
        ]
        recommendations.extend(
            category_posts.nlargest(10 // len(user_categories), 'rating_percent')['post_id'].tolist()
        )
    
    return list(dict.fromkeys(recommendations))[:10]

def get_collaborative_recommendations(df: pd.DataFrame, username: str) -> List[int]:
    user_item_matrix = df.pivot_table(
        index='username',
        columns='post_id',
        values='rating_percent',
        fill_value=0
    )
    
    if username not in user_item_matrix.index:
        return []
    
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    similar_users = user_similarity_df[username].sort_values(ascending=False)[1:6].index
    user_posts = set(df[df['username'] == username]['post_id'])
    recommendations = []
    
    for similar_user in similar_users:
        user_recommendations = df[
            (df['username'] == similar_user) & 
            (~df['post_id'].isin(user_posts))
        ].nlargest(2, 'rating_percent')['post_id'].tolist()
        recommendations.extend(user_recommendations)
    
    return list(dict.fromkeys(recommendations))[:10]

def get_mood_based_recommendations(df: pd.DataFrame, mood: str) -> List[int]:
    mood_mapping = {
        'happy': ['Entertainment', 'Comedy'],
        'sad': ['Motivation', 'Self-Help'],
        'energetic': ['Sports', 'Dance'],
        'relaxed': ['Nature', 'Music'],
        'focused': ['Education', 'Technology']
    }
    
    categories = mood_mapping.get(mood.lower(), [])
    if not categories:
        return df.nlargest(10, 'view_count')['post_id'].unique().tolist()
    
    mood_posts = df[df['category_name'].isin(categories)]
    return mood_posts.nlargest(10, 'view_count')['post_id'].unique().tolist()

def get_recommendations(
    df: pd.DataFrame,
    username: str,
    category_id: Optional[int] = None,
    mood: Optional[str] = None
) -> Tuple[List[dict], Dict[str, float]]:
    
    user_prefs = get_user_preferences(df, username)
    
    content_recs = get_content_based_recommendations(df, user_prefs)
    collaborative_recs = get_collaborative_recommendations(df, username)
    recommendations = []
    
    if mood:
        mood_recs = get_mood_based_recommendations(df, mood)
        recommendations.extend(mood_recs[:4])
    
    recommendations.extend(content_recs[:3])
    recommendations.extend(collaborative_recs[:3])
    
    recommendations = list(dict.fromkeys(recommendations))
    
    if category_id is not None:
        recommendations = [
            post_id for post_id in recommendations
            if df[df['post_id'] == post_id]['category_id'].iloc[0] == category_id
        ]
    
    if len(recommendations) < 10:
        additional_posts = df[
            (~df['post_id'].isin(recommendations)) &
            (~df['post_id'].isin(user_prefs['post_id']))
        ]
        if category_id is not None:
            additional_posts = additional_posts[additional_posts['category_id'] == category_id]
        
        additional_recommendations = additional_posts.nlargest(
            10 - len(recommendations),
            'rating_percent'
        )['post_id'].tolist()
        recommendations.extend(additional_recommendations)
    
    final_recommendations = []
    seen_posts = set()
    
    for post_id in recommendations[:10]: 
        if post_id in seen_posts:
            continue
            
        post_data = df[df['post_id'] == post_id].iloc[0]
        final_recommendations.append({
            'post_id': int(post_id),
            'category_name': post_data['category_name'],
            'thumbnail_url': post_data['thumbnail_url'],
            'video_link': post_data['video_link'],
            'view_count': float(post_data['view_count']),
            'rating_percent': float(post_data['rating_percent'])
        })
        
        seen_posts.add(post_id)
        
        if len(final_recommendations) >= 10:
            break
    
    evaluation_metrics = evaluate_recommendations(df)
    
    return final_recommendations, evaluation_metrics

@app.get("/feed")
async def get_feed(
    username: str,
    category_id: Optional[int] = None,
    mood: Optional[str] = None
):
    try:
        df = load_data()
        recommendations, metrics_results = get_recommendations(
            df=df,
            username=username,
            category_id=category_id,
            mood=mood
        )
        
        return {
            "recommendations": recommendations[:10],  # Ensure exactly 10 posts
            "metrics": metrics_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)