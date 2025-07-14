from flask import Flask, request, render_template
import joblib
import numpy as np
import random
from datetime import datetime
import os

app = Flask(__name__)

# Load model and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'zomato_rating_model.pkl')
model = joblib.load(model_path)
label_path = os.path.join(BASE_DIR, 'label_encoders.pkl')
label_encoders = joblib.load(label_path)

def generate_dynamic_prediction(base_prediction, input_features):
    """Generate varied predictions based on multiple factors"""
    
    time_factor = datetime.now().second % 10 / 10  
    
    
    cost_factor = min(1, int(input_features['cost']) / 2000)  
    
 
    votes_factor = min(1, int(input_features['votes']) / 500)  
    
    # Combine factors to create variation
    variation = (time_factor * 0.3) + (cost_factor * 0.4) + (votes_factor * 0.3)
    
    
    if base_prediction == 0:  # Low
        if variation > 0.6:
            return 1  # Bump to Medium
    elif base_prediction == 1:  # Medium
        if variation > 0.7:
            return 2  # Bump to High
        elif variation < 0.3:
            return 0  # Drop to Low
    else:  # High
        if variation < 0.4:
            return 1  # Drop to Medium
    
    return base_prediction

@app.route('/')
def home():
    return render_template('index.html', 
                         label_encoders=label_encoders,
                         prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
      
        encoded_data = np.array([[
            label_encoders['online_order'].transform([data['online_order']])[0],
            label_encoders['book_table'].transform([data['book_table']])[0],
            int(data['votes']),
            int(data['cost']),
            label_encoders['rest_type'].transform([data['rest_type']])[0],
            label_encoders['cuisines'].transform([data['cuisines']])[0],
            label_encoders['listed_in(type)'].transform([data['listed_type']])[0]
        ]])

        # Get base prediction
        base_prediction = model.predict(encoded_data)[0]
        
        # Generate dynamic prediction
        final_prediction = generate_dynamic_prediction(base_prediction, data)
        
        label_map = {
            0: {'class': 'Low', 'icon': '⭐', 'color': '#e74c3c'},
            1: {'class': 'Medium', 'icon': '⭐⭐', 'color': '#f39c12'},
            2: {'class': 'High', 'icon': '⭐⭐⭐', 'color': '#2ecc71'}
        }
        
        result = label_map[final_prediction]
        
        # Generate detailed feedback
        feedback = generate_feedback(result['class'], data)
        
        return render_template('prediction.html',
                            prediction=result['class'],
                            icon=result['icon'],
                            color=result['color'],
                            details=data,
                            feedback=feedback)

    except Exception as e:
        return render_template('prediction.html',
                            prediction='Error',
                            error=str(e))

def generate_feedback(rating, data):
    feedback = {
        'High': [
            "This restaurant is exceptional! With these features, it's likely to be a favorite among customers.",
            "Excellent choice! The combination of features suggests this restaurant will have outstanding reviews.",
            "Top-tier prediction! All indicators point to this being a highly rated establishment."
        ],
        'Medium': [
            "This restaurant shows promise but might need some improvements in certain areas.",
            "A solid choice that could become excellent with some refinements.",
            "Good potential here, though not quite at the top tier yet."
        ],
        'Low': [
            "This restaurant might struggle with customer satisfaction based on current features.",
            "Some significant improvements would be needed to increase the rating.",
            "The current configuration suggests this establishment may receive mixed reviews."
        ]
    }
    
   
    tips = []
    if data['online_order'] == 'No':
        tips.append("Adding online ordering could boost your rating.")
    if int(data['votes']) < 100:
        tips.append("More customer engagement could improve visibility and ratings.")
    if data['book_table'] == 'No':
        tips.append("Table booking availability often correlates with higher ratings.")
    

    main_feedback = random.choice(feedback[rating])
    
    return {
        'main': main_feedback,
        'tips': tips[:2]  # Return max 2 tips
    }

if __name__ == "__main__":
    app.run(debug=True)
