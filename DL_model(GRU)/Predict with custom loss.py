"""
PREDICTION SCRIPT WITH CUSTOM LOSS SUPPORT
This handles models trained with Focal Loss or other custom functions
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
# DEFINE CUSTOM LOSS FUNCTIONS (needed for loading)
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss - needed to load models trained with this loss
    """
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
    """
    Focal Loss with Label Smoothing - for advanced models
    """
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

model = None
try:
    # Try loading primary model
    model = keras.models.load_model(
        'african_football_improved_final.keras',
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Primary model loaded successfully")
except Exception as e:
    print(f"❌ Error loading primary model: {e}")
    print("\n💡 Trying alternative models...")
    
    alternative_models = ['best_model_improved.keras', 'best_enhanced_model.keras', 'best_model.keras']
    for model_name in alternative_models:
        try:
            print(f"   Trying: {model_name}")
            model = keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)
            print(f"   ✅ Successfully loaded: {model_name}")
            break
        except:
            continue

if model is None:
    print("\n❌ CRITICAL ERROR: Could not load any model file!")
    exit()

# Recompile for inference
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load preprocessors
try:
    scaler = joblib.load('scaler_improved.pkl')
    label_encoder = joblib.load('label_encoder_improved.pkl')
    team_encoder = joblib.load('team_encoder_improved.pkl')
    print("✅ Preprocessors loaded successfully")
except Exception as e:
    print(f"❌ Error loading preprocessors: {e}")
    exit()

# ============================================================================
# MODEL INFO (FIXED SCALER LOGIC)
# ============================================================================

print("\n" + "="*80)
print("MODEL INFORMATION")
print("="*80)

# RobustScaler Fix: Check for 'center_' instead of 'mean_'
if hasattr(scaler, 'center_'):
    num_features = scaler.center_.shape[0]
elif hasattr(scaler, 'n_features_in_'):
    num_features = scaler.n_features_in_
else:
    num_features = "Unknown"

print(f"\n📊 Model Architecture:")
print(f"  • Input shape: {model.input_shape}")
print(f"  • Total parameters: {model.count_params():,}")
print(f"  • Number of layers: {len(model.layers)}")

print(f"\n📊 Preprocessor Info:")
print(f"  • Number of features: {num_features}")
print(f"  • Output classes: {label_encoder.classes_}")
print(f"  • Number of teams: {len(team_encoder.classes_)}")

# ============================================================================
# MAKE SAMPLE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("MAKING SAMPLE PREDICTION")
print("="*80)

# Get model input requirements
seq_length = model.input_shape[1]
n_features_needed = model.input_shape[2]

# Create sample data (random for demo)
X_sample = np.random.randn(seq_length, n_features_needed)
X_sample_scaled = scaler.transform(X_sample)
X_sample_input = X_sample_scaled.reshape(1, seq_length, n_features_needed)

print("\n🔮 Predicting probabilities...")
probabilities = model.predict(X_sample_input, verbose=0)[0]
classes = label_encoder.classes_

print("\n⚽ Sample Prediction Result:")
print("="*60)

results = {}
label_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

for i, cls in enumerate(classes):
    full_name = label_map.get(cls, cls)
    prob_pct = probabilities[i] * 100
    results[full_name] = prob_pct
    bar = '█' * int(probabilities[i] * 40)
    print(f"  {full_name:15s}: {prob_pct:5.1f}%  {bar}")

predicted_outcome = max(results, key=results.get)
confidence_score = max(results.values())

print("-" * 60)
print(f"🎯 PREDICTED OUTCOME: {predicted_outcome}")
print(f"💪 CONFIDENCE: {confidence_score:.1f}%")
print("="*60)

# Betting recommendation
if confidence_score > 70:
    print("\n💰 Recommendation: STRONG BET ✅")
elif confidence_score > 55:
    print("\n💰 Recommendation: MODERATE BET ⚠️")
else:
    print("\n💰 Recommendation: AVOID/SKIP ❌")

print("\n" + "="*80)
print("SYSTEM READY FOR PRODUCTION! 🚀")
print("="*80)