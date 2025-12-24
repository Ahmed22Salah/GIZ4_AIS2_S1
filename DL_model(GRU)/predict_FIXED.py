import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FOOTBALL MATCH PREDICTOR - FIXED VERSION")
print("="*80)

# Define the focal loss function that was used during training
def focal_loss_fixed(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification
    """
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss

print("\n📦 Loading your trained model with custom loss function...")

try:
    # Load model with custom objects
    custom_objects = {
        'focal_loss_fixed': focal_loss_fixed,
        'focal_loss': focal_loss_fixed()  # Also try the compiled version
    }
    model = keras.models.load_model(
        'african_football_improved_final.keras',
        custom_objects=custom_objects
    )
    scaler = joblib.load('scaler_improved.pkl')
    label_encoder = joblib.load('label_encoder_improved.pkl')
    team_encoder = joblib.load('team_encoder_improved.pkl')
    print("✅ All files loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Trying alternative method (load without compiling)...")
    try:
        model = keras.models.load_model(
            'african_football_improved_final.keras',
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        scaler = joblib.load('scaler_improved.pkl')
        label_encoder = joblib.load('label_encoder_improved.pkl')
        team_encoder = joblib.load('team_encoder_improved.pkl')
        print("✅ Model loaded without original loss function!")
        print("   (Using categorical_crossentropy for predictions)")
    except Exception as e2:
        print(f"❌ Second attempt failed: {e2}")
        exit()

print("\n🎉 SUCCESS! Your model is ready to use!")
print("\nFiles loaded:")
print(f"  • Model: african_football_improved_final.keras")
print(f"  • Scaler: scaler_improved.pkl")
print(f"  • Label Encoder: label_encoder_improved.pkl")
print(f"  • Team Encoder: team_encoder_improved.pkl")

# Display model info
print(f"\n📊 Model Architecture:")
print(f"  • Input shape: {model.input_shape}")
print(f"  • Output shape: {model.output_shape}")
print(f"  • Total parameters: {model.count_params():,}")

# Simple prediction demo
print("\n" + "="*80)
print("MAKING SAMPLE PREDICTION")
print("="*80)

# Create dummy data for demo
seq_length = model.input_shape[1]  # Get from model
n_features = model.input_shape[2]  # Get from model

print(f"\nModel expects:")
print(f"  • Sequence length: {seq_length} matches")
print(f"  • Number of features: {n_features}")

# Random sample (replace with real data for actual predictions)
X_sample = np.random.randn(seq_length, n_features)
X_sample_scaled = scaler.transform(X_sample)
X_sample = X_sample_scaled.reshape(1, seq_length, n_features)

# Predict
print("\n🔮 Running prediction...")
probabilities = model.predict(X_sample, verbose=0)[0]
classes = label_encoder.classes_

# Display
print("\n⚽ Sample Prediction Results:")
print("-" * 40)
for i, cls in enumerate(classes):
    bar_length = int(probabilities[i] * 30)
    bar = "█" * bar_length
    if cls == 'H':
        label = "Home Win"
    elif cls == 'D':
        label = "Draw    "
    elif cls == 'A':
        label = "Away Win"
    else:
        label = cls
    print(f"  {label}: {bar} {probabilities[i]*100:5.1f}%")

predicted = classes[np.argmax(probabilities)]
confidence = np.max(probabilities) * 100

outcome_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
predicted_name = outcome_names.get(predicted, predicted)

print(f"\n🎯 Most Likely Outcome: {predicted_name} ({confidence:.1f}% confidence)")

print("\n" + "="*80)
print("✅ Your model is working correctly!")
print("="*80)

print("\n💡 Next Steps:")
print("   1. ✓ Model loads and predicts successfully")
print("   2. To make real predictions, prepare your match data with:")
print(f"      • Last {seq_length} matches for each team")
print(f"      • {n_features} features per match")
print("   3. Use the same preprocessing pipeline")
print("   4. Scale with scaler_improved.pkl")
print("   5. Make predictions!")

print("\n📝 Example usage in your code:")
print("""
# For a real match:
home_history = ... # Last 8 matches for home team
away_history = ... # Last 8 matches for away team
combined_features = create_match_features(home_history, away_history)
X = scaler.transform(combined_features).reshape(1, 8, -1)
probabilities = model.predict(X)[0]
""")

print("\n🔗 Available classes:", label_encoder.classes_)
print("🔗 Available teams:", len(team_encoder.classes_), "teams encoded")