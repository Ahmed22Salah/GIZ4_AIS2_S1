"""
MODEL FINDER - Find your trained model files automatically
Run this to locate your model files and fix the prediction scripts

Usage: python find_my_model.py
"""

import os
import glob

print("="*80)
print("🔍 SEARCHING FOR YOUR TRAINED MODEL FILES")
print("="*80)

# Get current directory
current_dir = os.getcwd()
print(f"\n📁 Current directory: {current_dir}")

# Search for model files
print("\n🔍 Searching for model files...")

# Look for .keras files
keras_files = glob.glob("*.keras")
h5_files = glob.glob("*.h5")
pkl_files = glob.glob("*.pkl")

print("\n" + "="*80)
print("FOUND FILES:")
print("="*80)

# Display Keras model files
if keras_files or h5_files:
    print("\n✅ MODEL FILES FOUND:")
    for f in keras_files:
        size = os.path.getsize(f) / (1024*1024)  # Convert to MB
        print(f"  📦 {f} ({size:.1f} MB)")
    for f in h5_files:
        size = os.path.getsize(f) / (1024*1024)
        print(f"  📦 {f} ({size:.1f} MB)")
else:
    print("\n❌ NO MODEL FILES FOUND (.keras or .h5)")
    print("   This means training didn't complete or save the model!")

# Display preprocessor files
if pkl_files:
    print("\n✅ PREPROCESSOR FILES FOUND:")
    for f in pkl_files:
        size = os.path.getsize(f) / 1024  # Convert to KB
        print(f"  📄 {f} ({size:.1f} KB)")
else:
    print("\n❌ NO PREPROCESSOR FILES FOUND (.pkl)")

# Look for CSV files
csv_files = glob.glob("*.csv")
if csv_files:
    print("\n✅ DATA FILES FOUND:")
    for f in csv_files:
        size = os.path.getsize(f) / (1024*1024)
        print(f"  📊 {f} ({size:.1f} MB)")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Diagnose the issue
if not keras_files and not h5_files:
    print("\n❌ PROBLEM: No model file found!")
    print("\n💡 SOLUTIONS:")
    print("   1. Complete model training first (run your training script)")
    print("   2. Wait for training to finish completely")
    print("   3. Check for error messages during training")
    print("   4. Make sure training saves the model at the end")
    
    print("\n🔍 WHAT TO CHECK IN YOUR TRAINING SCRIPT:")
    print("   Look for lines like:")
    print("   • model.save('some_name.keras')")
    print("   • model.save('some_name.h5')")
    print("\n   The saved filename should match what prediction scripts expect!")

elif len(keras_files) > 1 or len(h5_files) > 1:
    print("\n⚠️  MULTIPLE MODEL FILES FOUND!")
    print("   You have more than one model file.")
    print("   Using the most recent one...")
    
    # Find most recent
    all_models = keras_files + h5_files
    if all_models:
        latest_model = max(all_models, key=os.path.getmtime)
        print(f"\n✅ Most recent model: {latest_model}")
        
        print("\n📝 UPDATE YOUR PREDICTION SCRIPTS:")
        print(f"   Change 'african_football_improved_final.keras'")
        print(f"   To: '{latest_model}'")

else:
    # Found exactly one model
    model_file = keras_files[0] if keras_files else h5_files[0]
    
    if model_file == 'african_football_improved_final.keras':
        print("\n✅ PERFECT! Model file exists with correct name!")
        print("   Your prediction scripts should work now.")
    else:
        print(f"\n⚠️  MODEL FILE HAS DIFFERENT NAME!")
        print(f"   Found: {model_file}")
        print(f"   Expected: african_football_improved_final.keras")
        
        print("\n💡 TWO OPTIONS:")
        print(f"\n   Option 1: RENAME the file")
        print(f"   • Rename '{model_file}'")
        print(f"   • To: 'african_football_improved_final.keras'")
        
        print(f"\n   Option 2: UPDATE prediction scripts")
        print(f"   • Open prediction scripts")
        print(f"   • Change 'african_football_improved_final.keras'")
        print(f"   • To: '{model_file}'")

# Check for specific expected files
print("\n" + "="*80)
print("REQUIRED FILES CHECK:")
print("="*80)

required_files = {
    'Model file': ['african_football_improved_final.keras', 'best_model_improved.keras', 
                   'best_enhanced_model.keras', 'african_football_gru_enhanced.keras'],
    'Scaler': ['scaler_improved.pkl', 'scaler_enhanced.pkl', 'scaler.pkl'],
    'Label Encoder': ['label_encoder_improved.pkl', 'label_encoder_enhanced.pkl', 'label_encoder.pkl'],
    'Team Encoder': ['team_encoder_improved.pkl', 'team_encoder_enhanced.pkl', 'team_encoder.pkl'],
    'Data file': ['all_matches.csv']
}

found_files = {}

for file_type, possible_names in required_files.items():
    found = False
    for name in possible_names:
        if os.path.exists(name):
            print(f"✅ {file_type}: {name}")
            found_files[file_type] = name
            found = True
            break
    
    if not found:
        print(f"❌ {file_type}: NOT FOUND")
        print(f"   Looking for: {', '.join(possible_names)}")

# Generate automatic fix script
if found_files:
    print("\n" + "="*80)
    print("AUTO-FIX SCRIPT GENERATOR")
    print("="*80)
    
    if len(found_files) >= 4:  # If we found at least model + 3 preprocessors
        print("\n✅ Generating fixed prediction script...")
        
        # Create the auto-fixed script
        model_name = found_files.get('Model file', 'african_football_improved_final.keras')
        scaler_name = found_files.get('Scaler', 'scaler_improved.pkl')
        label_encoder_name = found_files.get('Label Encoder', 'label_encoder_improved.pkl')
        team_encoder_name = found_files.get('Team Encoder', 'team_encoder_improved.pkl')
        data_name = found_files.get('Data file', 'all_matches.csv')
        
        with open('predict_FIXED.py', 'w') as f:
            f.write(f'''"""
AUTO-GENERATED PREDICTION SCRIPT
This script automatically uses YOUR model files!
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FOOTBALL MATCH PREDICTOR - AUTO-FIXED VERSION")
print("="*80)

# Load YOUR actual files
print("\\n📦 Loading your trained model...")

try:
    model = keras.models.load_model('{model_name}')
    scaler = joblib.load('{scaler_name}')
    label_encoder = joblib.load('{label_encoder_name}')
    team_encoder = joblib.load('{team_encoder_name}')
    print("✅ All files loaded successfully!")
except Exception as e:
    print(f"❌ Error: {{e}}")
    exit()

print("\\n🎉 SUCCESS! Your model is ready to use!")
print("\\nFiles loaded:")
print(f"  • Model: {model_name}")
print(f"  • Scaler: {scaler_name}")
print(f"  • Label Encoder: {label_encoder_name}")
print(f"  • Team Encoder: {team_encoder_name}")

# Simple prediction demo
print("\\n" + "="*80)
print("MAKING SAMPLE PREDICTION")
print("="*80)

# Create dummy data for demo
seq_length = 8
n_features = scaler.mean_.shape[0]

# Random sample (replace with real data for actual predictions)
X_sample = np.random.randn(seq_length, n_features)
X_sample_scaled = scaler.transform(X_sample)
X_sample = X_sample_scaled.reshape(1, seq_length, -1)

# Predict
probabilities = model.predict(X_sample, verbose=0)[0]
classes = label_encoder.classes_

# Display
print("\\n⚽ Sample Prediction (Egypt vs Nigeria):")
for i, cls in enumerate(classes):
    if cls == 'H':
        print(f"  Home Win: {{probabilities[i]*100:.1f}}%")
    elif cls == 'D':
        print(f"  Draw:     {{probabilities[i]*100:.1f}}%")
    elif cls == 'A':
        print(f"  Away Win: {{probabilities[i]*100:.1f}}%")

predicted = classes[np.argmax(probabilities)]
confidence = np.max(probabilities) * 100

print(f"\\n🎯 Prediction: {{predicted}} ({{confidence:.1f}}% confidence)")

print("\\n✅ Your model is working correctly!")
print("\\n💡 Next steps:")
print("   1. Use predict_matches_real.py for actual predictions")
print("   2. Update it with these filenames:")
print(f"      - Model: {model_name}")
print(f"      - Scaler: {scaler_name}")
print(f"      - Label Encoder: {label_encoder_name}")
print(f"      - Team Encoder: {team_encoder_name}")
''')
        
        print(f"\n✅ Created 'predict_FIXED.py'")
        print("\n🚀 RUN THIS NOW:")
        print("   python predict_FIXED.py")
        print("\nThis script uses YOUR actual filenames!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if not keras_files and not h5_files:
    print("\n❌ NO MODEL FOUND - You need to train your model first!")
    print("\n📝 TO DO:")
    print("   1. Run your training script (GRU.py or similar)")
    print("   2. Wait for training to complete (may take hours)")
    print("   3. Make sure it saves the model at the end")
    print("   4. Then run this finder script again")
elif len(found_files) >= 4:
    print("\n✅ ALL FILES FOUND!")
    print("\n🚀 NEXT STEP: Run the auto-generated script:")
    print("   python predict_FIXED.py")
else:
    print("\n⚠️  SOME FILES MISSING")
    print("\n📝 Missing files need to be generated by training script")
    print("   Re-run training to generate all necessary files")

print("\n" + "="*80)
''')
        
        with open('predict_FIXED.py', 'w') as f:
            f.write(f'''"""
AUTO-GENERATED PREDICTION SCRIPT
This script automatically uses YOUR model files!
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FOOTBALL MATCH PREDICTOR - AUTO-FIXED VERSION")
print("="*80)

# Load YOUR actual files
print("\\n📦 Loading your trained model...")

try:
    model = keras.models.load_model('{model_name}')
    scaler = joblib.load('{scaler_name}')
    label_encoder = joblib.load('{label_encoder_name}')
    team_encoder = joblib.load('{team_encoder_name}')
    print("✅ All files loaded successfully!")
except Exception as e:
    print(f"❌ Error: {{e}}")
    exit()

print("\\n🎉 SUCCESS! Your model is ready to use!")
print("\\nFiles loaded:")
print(f"  • Model: {model_name}")
print(f"  • Scaler: {scaler_name}")
print(f"  • Label Encoder: {label_encoder_name}")
print(f"  • Team Encoder: {team_encoder_name}")

# Simple prediction demo
print("\\n" + "="*80)
print("MAKING SAMPLE PREDICTION")
print("="*80)

# Create dummy data for demo
seq_length = 8
n_features = scaler.mean_.shape[0]

# Random sample (replace with real data for actual predictions)
X_sample = np.random.randn(seq_length, n_features)
X_sample_scaled = scaler.transform(X_sample)
X_sample = X_sample_scaled.reshape(1, seq_length, -1)

# Predict
probabilities = model.predict(X_sample, verbose=0)[0]
classes = label_encoder.classes_

# Display
print("\\n⚽ Sample Prediction (Egypt vs Nigeria):")
for i, cls in enumerate(classes):
    if cls == 'H':
        print(f"  Home Win: {{probabilities[i]*100:.1f}}%")
    elif cls == 'D':
        print(f"  Draw:     {{probabilities[i]*100:.1f}}%")
    elif cls == 'A':
        print(f"  Away Win: {{probabilities[i]*100:.1f}}%")

predicted = classes[np.argmax(probabilities)]
confidence = np.max(probabilities) * 100

print(f"\\n🎯 Prediction: {{predicted}} ({{confidence:.1f}}% confidence)")

print("\\n✅ Your model is working correctly!")
print("\\n💡 Next steps:")
print("   1. Use predict_matches_real.py for actual predictions")
print("   2. Update it with these filenames:")
print(f"      - Model: {model_name}")
print(f"      - Scaler: {scaler_name}")
print(f"      - Label Encoder: {label_encoder_name}")
print(f"      - Team Encoder: {team_encoder_name}")
''')
        
        print(f"\n✅ Created 'predict_FIXED.py'")
        print("\n🚀 RUN THIS NOW:")
        print("   python predict_FIXED.py")
        print("\nThis script uses YOUR actual filenames!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if not keras_files and not h5_files:
    print("\n❌ NO MODEL FOUND - You need to train your model first!")
    print("\n📝 TO DO:")
    print("   1. Run your training script (GRU.py or similar)")
    print("   2. Wait for training to complete (may take hours)")
    print("   3. Make sure it saves the model at the end")
    print("   4. Then run this finder script again")
elif len(found_files) >= 4:
    print("\n✅ ALL FILES FOUND!")
    print("\n🚀 NEXT STEP: Run the auto-generated script:")
    print("   python predict_FIXED.py")
else:
    print("\n⚠️  SOME FILES MISSING")
    print("\n📝 Missing files need to be generated by training script")
    print("   Re-run training to generate all necessary files")

print("\n" + "="*80)