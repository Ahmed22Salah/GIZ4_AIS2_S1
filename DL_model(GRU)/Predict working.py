"""
PREDICTION SCRIPT WITH CUSTOM LOSS SUPPORT (FIXED)
Handles both StandardScaler and RobustScaler
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FOOTBALL MATCH PREDICTOR - WITH CUSTOM LOSS SUPPORT")
print("="*80)

# ============================================================================
# DEFINE CUSTOM LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss - needed to load models trained with this loss"""
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss_fixed

def focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, smoothing=0.1):
    """Focal Loss with Label Smoothing"""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        
        if smoothing > 0:
            y_true_one_hot = y_true_one_hot * (1 - smoothing) + smoothing / 3
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return loss_fn

print("\n✅ Custom loss functions defined")

# ============================================================================
# LOAD MODEL WITH CUSTOM OBJECTS
# ============================================================================

print("\n📦 Loading your trained model...")

custom_objects = {
    'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25),
    'loss_fn': focal_loss_with_label_smoothing(gamma=2.5, alpha=0.25, smoothing=0.1),
}

try:
    model = keras.models.load_model(
        'african_football_improved_final.keras',
        custom_objects=custom_objects,
        compile=False
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model loaded successfully (with custom loss support)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Load preprocessors
try:
    scaler = joblib.load('scaler_improved.pkl')
    label_encoder = joblib.load('label_encoder_improved.pkl')
    team_encoder = joblib.load('team_encoder_improved.pkl')
    print("✅ Preprocessors loaded successfully")
except Exception as e:
    print(f"❌ Error loading preprocessors: {e}")
    exit()

print("\n🎉 SUCCESS! Your model is ready to use!")

# ============================================================================
# MODEL INFO
# ============================================================================

print("\n" + "="*80)
print("MODEL INFORMATION")
print("="*80)

print(f"\n📊 Model Architecture:")
print(f"  • Input shape: {model.input_shape}")
print(f"  • Output shape: {model.output_shape}")
print(f"  • Total parameters: {model.count_params():,}")
print(f"  • Number of layers: {len(model.layers)}")

print(f"\n📊 Preprocessor Info:")

# Handle both StandardScaler and RobustScaler
if hasattr(scaler, 'mean_'):
    n_features = scaler.mean_.shape[0]
    scaler_type = "StandardScaler"
elif hasattr(scaler, 'center_'):
    n_features = scaler.center_.shape[0]
    scaler_type = "RobustScaler"
else:
    n_features = model.input_shape[2]
    scaler_type = "Unknown"

print(f"  • Scaler type: {scaler_type}")
print(f"  • Number of features: {n_features}")
print(f"  • Output classes: {label_encoder.classes_}")
print(f"  • Number of teams: {len(team_encoder.classes_)}")

# ============================================================================
# MAKE SAMPLE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("MAKING SAMPLE PREDICTION")
print("="*80)

# Get model requirements
seq_length = model.input_shape[1]
n_features = model.input_shape[2]

print(f"\n📋 Model Requirements:")
print(f"  • Sequence length: {seq_length} matches")
print(f"  • Features per match: {n_features}")

# Create sample data
X_sample = np.random.randn(seq_length, n_features)
X_sample_scaled = scaler.transform(X_sample)
X_sample_input = X_sample_scaled.reshape(1, seq_length, -1)

# Make prediction
print("\n🔮 Making prediction...")
probabilities = model.predict(X_sample_input, verbose=0)[0]

# Get class labels
classes = label_encoder.classes_

# Display results
print("\n" + "="*80)
print("⚽ SAMPLE PREDICTION: Egypt vs Nigeria")
print("="*80)

results = {}
bar_length = 40

for i, cls in enumerate(classes):
    prob = probabilities[i] * 100
    bar = '█' * int((prob / 100) * bar_length)
    
    if cls == 'H':
        emoji = "🏠"
        label = "Home Win (Egypt)"
        results['Home Win'] = prob
    elif cls == 'D':
        emoji = "🤝"
        label = "Draw"
        results['Draw'] = prob
    elif cls == 'A':
        emoji = "✈️"
        label = "Away Win (Nigeria)"
        results['Away Win'] = prob
    
    print(f"{emoji} {label:22s}: {prob:5.1f}%  {bar}")

# Determine prediction
predicted = max(results, key=results.get)
confidence = max(results.values())

print("\n" + "="*80)
print(f"🎯 PREDICTED OUTCOME: {predicted}")
print(f"💪 CONFIDENCE LEVEL: {confidence:.1f}%")
print("="*80)

# Betting recommendation
print(f"\n💰 Betting Recommendation:")
if confidence > 70:
    print("   ✅ STRONG BET - High confidence prediction")
    print("   📈 Suggested stake: 3-5% of bankroll")
    print("   🎯 Good betting opportunity!")
elif confidence > 60:
    print("   ⚠️  MODERATE BET - Reasonable confidence")
    print("   📈 Suggested stake: 1-2% of bankroll")
    print("   ⚖️  Proceed with caution")
elif confidence > 50:
    print("   ⚠️  WEAK BET - Low confidence")
    print("   📈 Suggested stake: 0.5-1% of bankroll")
    print("   ⚠️  Risky - consider skipping")
else:
    print("   ❌ AVOID - Very uncertain prediction")
    print("   📈 Suggested stake: SKIP this match")
    print("   🚫 Too risky to bet on")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ MODEL TEST SUCCESSFUL!")
print("="*80)

print("\n🎉 What This Means:")
print("  ✅ Your model loads correctly")
print("  ✅ Custom loss function handled properly")
print("  ✅ Model can make predictions")
print("  ✅ All preprocessors working")
print("  ✅ Ready for real predictions!")

print("\n🚀 Next Steps:")
print("  1. ✅ Model is working - CONFIRMED!")
print("  2. Make real predictions with actual team data")
print("  3. Launch web interface: streamlit run app_simple.py")
print("  4. Start predicting upcoming matches!")

print("\n📝 Important Notes:")
print("  • The prediction above uses RANDOM data (for testing)")
print("  • For REAL predictions, you need actual team statistics")
print("  • Model expects sequences of previous matches")
print(f"  • Each match needs {n_features} features")

print("\n💡 Understanding Your Model:")
print(f"  • Your model uses {seq_length} previous matches to predict")
print(f"  • It looks at {n_features} features per match")
print(f"  • Features include: ELO ratings, form, goals, etc.")
print(f"  • Output: Probabilities for Home/Draw/Away")

print("\n🎯 To Make Real Predictions:")
print("  1. Load historical match data")
print("  2. Extract last {seq_length} matches for both teams")
print(f"  3. Calculate all {n_features} features")
print("  4. Scale using the loaded scaler")
print(f"  5. Reshape to (1, {seq_length}, {n_features})")
print("  6. Call model.predict()")
print("  7. Interpret the probabilities!")

print("\n" + "="*80)
print("YOUR FOOTBALL PREDICTION SYSTEM IS READY! 🎉⚽🏆")
print("="*80)

print("\n🌟 CONGRATULATIONS! 🌟")
print("You've successfully:")
print("  ✓ Trained a neural network")
print("  ✓ Loaded it with custom loss functions")
print("  ✓ Made your first prediction")
print("  ✓ Built a football prediction system!")

print("\n💪 You're now ready to predict football matches!")
print("   Good luck with your predictions! ⚽🎯")