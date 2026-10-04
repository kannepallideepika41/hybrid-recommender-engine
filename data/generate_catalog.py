import os
import numpy as np
import pandas as pd

def generate_ecommerce_data(n_products=1000, n_interactions=50000, random_state=42):
    np.random.seed(random_state)
    
    categories = ['Electronics', 'Home & Kitchen', 'Footwear', 'Apparel', 'Fitness', 'Books']
    adjectives = ['Ultra', 'Smart', 'Ergonomic', 'Wireless', 'Premium', 'Eco-Friendly', 'Pro']
    nouns = ['Device', 'Sneakers', 'Blender', 'Tracker', 'Headphones', 'Backpack', 'Jacket']
    
    # 1. Product Catalog
    products = []
    for i in range(1, n_products + 1):
        cat = np.random.choice(categories)
        title = f"{np.random.choice(adjectives)} {np.random.choice(nouns)} {i}"
        price = np.round(np.random.uniform(15.0, 450.0), 2)
        rating = np.round(np.random.uniform(2.5, 5.0), 1)
        popularity_score = np.random.poisson(lam=25)
        
        products.append({
            'product_id': f"PROD_{i:04d}",
            'title': title,
            'category': cat,
            'price': price,
            'rating': rating,
            'popularity_score': popularity_score
        })
    df_products = pd.DataFrame(products)
    
    # 2. User Interactions & Click Logs (Ranking Dataset)
    user_ids = [f"USER_{np.random.randint(1, 2000):04d}" for _ in range(n_interactions)]
    sampled_products = df_products.sample(n=n_interactions, replace=True).reset_index(drop=True)
    
    user_session_duration = np.random.exponential(scale=120, size=n_interactions).round(1)
    user_price_sensitivity = np.random.uniform(0.1, 1.0, size=n_interactions)
    price_diff_ratio = (sampled_products['price'] - 100) / 100
    
    # Click probability logic
    log_odds = (
        -1.5
        + 0.5 * (sampled_products['rating'] - 3.0)
        + 0.02 * sampled_products['popularity_score']
        - 1.2 * (user_price_sensitivity * np.maximum(0, price_diff_ratio))
        + 0.003 * user_session_duration
    )
    click_prob = 1 / (1 + np.exp(-log_odds))
    is_clicked = (np.random.uniform(0, 1, size=n_interactions) < click_prob).astype(int)
    
    df_interactions = pd.DataFrame({
        'user_id': user_ids,
        'product_id': sampled_products['product_id'],
        'category': sampled_products['category'],
        'price': sampled_products['price'],
        'rating': sampled_products['rating'],
        'popularity_score': sampled_products['popularity_score'],
        'user_session_duration': user_session_duration,
        'user_price_sensitivity': np.round(user_price_sensitivity, 2),
        'is_clicked': is_clicked
    })
    
    os.makedirs('data', exist_ok=True)
    df_products.to_csv('data/products.csv', index=False)
    df_interactions.to_csv('data/interactions.csv', index=False)
    print(f"Generated {n_products} catalog items and {n_interactions} interaction events.")

if __name__ == '__main__':
    generate_ecommerce_data()