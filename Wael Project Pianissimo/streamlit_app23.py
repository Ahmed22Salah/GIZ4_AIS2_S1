import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================
@st.cache_resource
def load_models():
    dt_model = joblib.load('models/decision_tree_model.pkl')
    nb_model = joblib.load('models/naive_bayes_model.pkl')
    nn_model = joblib.load('models/neural_network_model.pkl')
    lr_model = joblib.load('models/linear_regression_model.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    
    scaler_class = joblib.load('models/scaler_class.pkl')
    scaler_reg = joblib.load('models/scaler_reg.pkl')
    scaler_kmeans = joblib.load('models/scaler_kmeans.pkl')
    
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    
    return {
        'models': {
            'Decision Tree': dt_model,
            'Naive Bayes': nb_model,
            'Neural Network': nn_model,
            'Linear Regression': lr_model,
            'K-Means': kmeans_model
        },
        'scalers': {
            'class': scaler_class,
            'reg': scaler_reg,
            'kmeans': scaler_kmeans
        },
        'label_encoders': label_encoders,
        'feature_info': feature_info
    }


def prepare_label_encoded_input(input_data, feature_info, label_encoders):
    """Prepare input for Decision Tree (label encoded)."""

    # Feature columns for classification (label-encoded version)
    feature_cols = [
        'Age', 'Gender', 'Academic_Level', 'Country',
        'Avg_Daily_Usage_Hours', 'Most_Used_Platform',
        'Sleep_Hours_Per_Night', 'Mental_Health_Score',
        'Relationship_Status', 'Conflicts_Over_Social_Media',
        'Region', 'Sleep_Deficit', 'Usage_Sleep_Ratio',
        'Relationship_Strain', 'Addiction_Risk_Score'
    ]

    # Create DataFrame with one row
    df_input = pd.DataFrame([input_data])

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Keep only the feature columns in the correct order
    df_input = df_input[feature_cols]

    # Apply label encoding to categorical columns
    categorical_cols = ['Gender', 'Academic_Level', 'Country',
                        'Most_Used_Platform', 'Relationship_Status', 'Region']

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels by using a default value
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError:
                # If the value wasn't seen during training, use the first class
                df_input[col] = 0

    return df_input
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_engineered_features(input_data):
    """Create the engineered features from raw input."""
    # Sleep Deficit
    input_data['Sleep_Deficit'] = max(0, 8 - input_data['Sleep_Hours_Per_Night'])
    
    # Usage-Sleep Ratio
    input_data['Usage_Sleep_Ratio'] = input_data['Avg_Daily_Usage_Hours'] / (input_data['Sleep_Hours_Per_Night'] + 0.1)
    
    # Relationship Strain
    relationship_weights = {'Single': 1.0, 'In Relationship': 1.5, 'Complicated': 2.0}
    input_data['Relationship_Strain'] = (
        input_data['Conflicts_Over_Social_Media'] * 
        relationship_weights.get(input_data['Relationship_Status'], 1.0)
    )
    
    # Addiction Risk Score (simplified version for interface)
    # Normalize each component to 0-1 range using expected ranges
    usage_norm = min(1, input_data['Avg_Daily_Usage_Hours'] / 10)
    sleep_deficit_norm = min(1, input_data['Sleep_Deficit'] / 4)
    conflict_norm = min(1, input_data['Conflicts_Over_Social_Media'] / 10)
    mental_health_norm = 1 - (input_data['Mental_Health_Score'] / 10)
    
    input_data['Addiction_Risk_Score'] = (
        0.35 * usage_norm +
        0.25 * sleep_deficit_norm +
        0.20 * conflict_norm +
        0.20 * mental_health_norm
    )
    
    return input_data


def prepare_onehot_encoded_input(input_data, feature_info, scaler):
    """Prepare input for Naive Bayes and Neural Network (one-hot encoded)."""

    # Get the exact column names the model was trained on
    expected_cols = feature_info['X_class_columns']

    # Create DataFrame with all expected columns, initialized to 0
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)

    # Get numeric column names
    numeric_cols = feature_info['numeric_cols']

    # Fill numeric columns with actual values
    for col in numeric_cols:
        if col in df_input.columns and col in input_data:
            df_input.loc[0, col] = input_data[col]

    # Fill one-hot encoded categorical columns
    categorical_mappings = {
        'Gender': input_data.get('Gender', 'Male'),
        'Academic_Level': input_data.get('Academic_Level', 'Undergraduate'),
        'Country': input_data.get('Country', 'USA'),
        'Most_Used_Platform': input_data.get('Most_Used_Platform', 'Instagram'),
        'Relationship_Status': input_data.get('Relationship_Status', 'Single'),
        'Region': input_data.get('Region', 'Northern America')
    }

    for cat_col, value in categorical_mappings.items():
        # Try different naming conventions
        possible_col_names = [
            f"{cat_col}_{value}",
            f"{cat_col}_{value}".replace(" ", "_"),
        ]

        for col_name in possible_col_names:
            if col_name in df_input.columns:
                df_input.loc[0, col_name] = 1
                break

    # Scale numeric features
    numeric_cols_in_df = [col for col in numeric_cols if col in df_input.columns]

    if len(numeric_cols_in_df) > 0:
        numeric_data = df_input[numeric_cols_in_df].copy()
        df_input[numeric_cols_in_df] = scaler.transform(numeric_data)

    return df_input

def prepare_regression_input(input_data, feature_info, scaler):
    """Prepare input for regression model."""

    expected_cols = feature_info['X_reg_columns']
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)

    numeric_cols = feature_info['numeric_cols']

    # Fill numeric columns
    for col in numeric_cols:
        if col in df_input.columns and col in input_data:
            df_input.loc[0, col] = input_data[col]

    # Fill one-hot encoded categorical columns
    categorical_mappings = {
        'Gender': input_data.get('Gender', 'Male'),
        'Academic_Level': input_data.get('Academic_Level', 'Undergraduate'),
        'Country': input_data.get('Country', 'USA'),
        'Most_Used_Platform': input_data.get('Most_Used_Platform', 'Instagram'),
        'Relationship_Status': input_data.get('Relationship_Status', 'Single'),
        'Region': input_data.get('Region', 'Northern America'),
        'Affects_Academic_Performance': input_data.get('Affects_Academic_Performance', 'No')
    }

    for cat_col, value in categorical_mappings.items():
        possible_col_names = [
            f"{cat_col}_{value}",
            f"{cat_col}_{value}".replace(" ", "_"),
        ]

        for col_name in possible_col_names:
            if col_name in df_input.columns:
                df_input.loc[0, col_name] = 1
                break

    # Scale numeric features
    numeric_cols_in_df = [col for col in numeric_cols if col in df_input.columns]

    if len(numeric_cols_in_df) > 0:
        numeric_data = df_input[numeric_cols_in_df].copy()
        df_input[numeric_cols_in_df] = scaler.transform(numeric_data)

    return df_input

def prepare_kmeans_input(input_data, feature_info, scaler):
    """Prepare input for K-Means clustering."""
    kmeans_cols = feature_info['kmeans_columns']
    df_input = pd.DataFrame([input_data])[kmeans_cols]
    df_input_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=kmeans_cols
    )
    return df_input_scaled

def get_cluster_profile(cluster_id):
    """Return a description of each cluster."""
    profiles = {
        0: ("🟢 Low Risk", "Healthy balance between social media use and life. Good sleep, low addiction indicators."),
        1: ("🟡 Moderate Risk", "Some signs of overuse. May need to monitor habits and set boundaries."),
        2: ("🔴 High Risk", "High usage, poor sleep, elevated addiction markers. Consider intervention."),
        3: ("🟠 At-Risk", "Borderline patterns. Early intervention recommended.")
    }
    return profiles.get(cluster_id, ("❓ Unknown", "Cluster profile not defined."))

def create_gauge_chart(value, title, max_val=10):
    """Create a gauge chart for visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.33], 'color': "lightgreen"},
                {'range': [max_val*0.33, max_val*0.66], 'color': "yellow"},
                {'range': [max_val*0.66, max_val], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_risk_radar(input_data):
    """Create a radar chart showing risk factors."""
    categories = ['Usage', 'Sleep Deficit', 'Conflicts', 'Mental Health (inv)', 'Relationship Strain']
    
    # Normalize values to 0-10 scale
    values = [
        min(10, input_data['Avg_Daily_Usage_Hours']),
        min(10, input_data['Sleep_Deficit'] * 2.5),
        min(10, input_data['Conflicts_Over_Social_Media']),
        10 - input_data['Mental_Health_Score'],
        min(10, input_data['Relationship_Strain'])
    ]
    values.append(values[0])  # Close the radar
    categories.append(categories[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load models
    try:
        artifacts = load_models()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.info("Please run the training script first to generate model files.")
        return
    
    models = artifacts['models']
    scalers = artifacts['scalers']
    feature_info = artifacts['feature_info']
    
    # ========== HEADER ==========
    st.title("📱 Social Media Addiction Predictor")
    st.markdown("""
    This application uses **Machine Learning** to predict:
    - Whether social media affects your **academic performance**
    - Your estimated **addiction score**
    - Your **risk profile** based on usage patterns
    
    ---
    """)
    
    # ========== SIDEBAR: INPUT FORM ==========
    st.sidebar.header("📝 Enter Your Information")
    
    # Personal Info
    st.sidebar.subheader("Personal Info")
    age = st.sidebar.slider("Age", 14, 35, 20)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
    academic_level = st.sidebar.selectbox(
        "Academic Level",
        ["High School", "Undergraduate", "Postgraduate"]
    )
    
    country = st.sidebar.selectbox(
        "Country",
        ["USA", "UK", "India", "Bangladesh", "Germany", "France", 
         "Australia", "Canada", "Pakistan", "Brazil", "Japan", "Other"]
    )
    
    # Map country to region
    country_to_region = {
        "USA": "Northern America",
        "UK": "Northern Europe",
        "India": "Southern Asia",
        "Bangladesh": "Southern Asia",
        "Germany": "Western Europe",
        "France": "Western Europe",
        "Australia": "Australia and New Zealand",
        "Canada": "Northern America",
        "Pakistan": "Southern Asia",
        "Brazil": "Latin America and the Caribbean",
        "Japan": "Eastern Asia",
        "Other": "Other"
    }
    region = country_to_region.get(country, "Other")
    
    # Social Media Usage
    st.sidebar.subheader("📲 Social Media Usage")
    avg_usage = st.sidebar.slider(
        "Average Daily Usage (hours)",
        0.0, 12.0, 3.0, 0.5
    )
    platform = st.sidebar.selectbox(
        "Most Used Platform",
        ["Instagram", "TikTok", "YouTube", "Facebook", "Twitter", 
         "Snapchat", "WhatsApp", "LinkedIn", "Reddit", "Other"]
    )
    
    # Health & Wellbeing
    st.sidebar.subheader("😴 Health & Wellbeing")
    sleep_hours = st.sidebar.slider(
        "Sleep Hours Per Night",
        3.0, 10.0, 7.0, 0.5
    )
    mental_health = st.sidebar.slider(
        "Mental Health Score (1-10)",
        1, 10, 7,
        help="1 = Very Poor, 10 = Excellent"
    )
    
    # Relationships
    st.sidebar.subheader("💑 Relationships")
    relationship_status = st.sidebar.selectbox(
        "Relationship Status",
        ["Single", "In Relationship", "Complicated"]
    )
    conflicts = st.sidebar.slider(
        "Conflicts Over Social Media (1-10)",
        1, 10, 3,
        help="How often does social media cause conflicts?"
    )
    # ==================== COLLECT INPUT DATA ====================
    # This MUST be BEFORE the if predict_button check!

    input_data = {
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Country': country,
        'Most_Used_Platform': platform,
        'Avg_Daily_Usage_Hours': usage_hours,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Mental_Health_Score': mental_health,
        'Relationship_Status': relationship_status,
        'Conflicts_Over_Social_Media': conflicts
    }

    # Add Region based on Country
    region_mapping = {
        'USA': 'Northern America',
        'Canada': 'Northern America',
        'UK': 'Northern Europe',
        'Germany': 'Western Europe',
        'France': 'Western Europe',
        'India': 'Southern Asia',
        'Japan': 'Eastern Asia',
        'South Korea': 'Eastern Asia',
        'Australia': 'Australia and New Zealand',
        'Brazil': 'South America',
        'Mexico': 'Central America',
        'UAE': 'Western Asia',
        'Other': 'Other'
    }
    input_data['Region'] = region_mapping.get(country, 'Other')
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)
    
    # ========== MAIN CONTENT ==========
    if predict_button:
        # Create engineered features
        input_data = create_engineered_features(input_data)

        # ==================== PREPARE INPUTS ====================

        # Label-encoded input for Decision Tree
        X_label = prepare_label_encoded_input(
            input_data,
            feature_info,
            models['label_encoders']
        )

        # One-hot encoded input for Naive Bayes and Neural Network
        X_onehot = prepare_onehot_encoded_input(
            input_data,
            feature_info,
            scalers['class']
        )

        # Regression input
        X_reg = prepare_regression_input(
            input_data,
            feature_info,
            scalers['reg']
        )

        # K-Means input
        kmeans_cols = feature_info['kmeans_columns']
        X_kmeans_data = {col: input_data.get(col, 0) for col in kmeans_cols if col != 'Addicted_Score'}
        X_kmeans = pd.DataFrame([X_kmeans_data])

        # Add placeholder for Addicted_Score (we'll predict it)
        X_kmeans['Addicted_Score'] = 5  # Placeholder, will be updated
        X_kmeans = X_kmeans[kmeans_cols]  # Ensure correct order
        X_kmeans_scaled = scalers['kmeans'].transform(X_kmeans)

        # ==================== MAKE PREDICTIONS ====================

        st.header("🎯 Prediction Results")

        # Classification predictions
        predictions = {}
        probabilities = {}

        # Decision Tree (uses label-encoded data)
        dt_pred = models['decision_tree'].predict(X_label)[0]
        dt_prob = models['decision_tree'].predict_proba(X_label)[0][1]
        predictions['Decision Tree'] = dt_pred
        probabilities['Decision Tree'] = dt_prob

        # Naive Bayes (uses one-hot encoded data)
        nb_pred = models['naive_bayes'].predict(X_onehot)[0]
        nb_prob = models['naive_bayes'].predict_proba(X_onehot)[0][1]
        predictions['Naive Bayes'] = nb_pred
        probabilities['Naive Bayes'] = nb_prob

        # Neural Network (uses one-hot encoded data)
        nn_pred = models['neural_network'].predict(X_onehot)[0]
        nn_prob = models['neural_network'].predict_proba(X_onehot)[0][1]
        predictions['Neural Network'] = nn_pred
        probabilities['Neural Network'] = nn_prob

        # Ensemble (majority vote)
        ensemble_pred = 1 if sum(predictions.values()) >= 2 else 0
        avg_prob = sum(probabilities.values()) / len(probabilities)

        # Linear Regression
        predicted_addiction = models['linear_regression'].predict(X_reg)[0]
        predicted_addiction = np.clip(predicted_addiction, 1, 10)  # Keep in valid range

        # K-Means Clustering
        # Update with predicted addiction score
        X_kmeans['Addicted_Score'] = predicted_addiction
        X_kmeans_scaled = scalers['kmeans'].transform(X_kmeans)
        cluster = models['kmeans'].predict(X_kmeans_scaled)[0]

        # ==================== DISPLAY RESULTS ====================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📚 Academic Impact")
            if ensemble_pred == 1:
                st.error("⚠️ **YES** - Social media likely affects your academics")
            else:
                st.success("✅ **NO** - Social media unlikely to affect your academics")

            st.metric("Risk Probability", f"{avg_prob * 100:.1f}%")

            with st.expander("Model Breakdown"):
                for name, prob in probabilities.items():
                    pred_text = "Yes" if predictions[name] == 1 else "No"
                    st.write(f"**{name}:** {pred_text} ({prob * 100:.1f}% risk)")

        with col2:
            st.subheader("📊 Addiction Score")
            st.metric("Predicted Score", f"{predicted_addiction:.1f} / 10")

            # Color based on severity
            if predicted_addiction <= 3:
                st.success("Low addiction level")
            elif predicted_addiction <= 6:
                st.warning("Moderate addiction level")
            else:
                st.error("High addiction level")

        with col3:
            st.subheader("👥 User Cluster")
            st.metric("Cluster", f"Group {cluster}")

            # Cluster descriptions (customize based on your analysis)
            cluster_descriptions = {
                0: "Balanced users with healthy habits",
                1: "Moderate users with some concerns",
                2: "Heavy users at risk",
                3: "High-risk users needing intervention"
            }
            desc = cluster_descriptions.get(cluster, "User profile identified")
            st.info(desc)

        # ==================== ADDITIONAL INSIGHTS ====================

        st.header("💡 Insights & Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Input Summary")
            st.write(f"**Daily Usage:** {input_data['Avg_Daily_Usage_Hours']:.1f} hours")
            st.write(f"**Sleep:** {input_data['Sleep_Hours_Per_Night']:.1f} hours")
            st.write(f"**Sleep Deficit:** {input_data['Sleep_Deficit']:.1f} hours below recommended")
            st.write(f"**Mental Health Score:** {input_data['Mental_Health_Score']}/10")
            st.write(f"**Addiction Risk Score:** {input_data['Addiction_Risk_Score']:.2f}")

        with col2:
            st.subheader("Recommendations")
            recommendations = []

            if input_data['Avg_Daily_Usage_Hours'] > 5:
                recommendations.append("🕐 Consider reducing daily social media usage to under 5 hours")

            if input_data['Sleep_Hours_Per_Night'] < 7:
                recommendations.append("😴 Try to get at least 7-8 hours of sleep")

            if input_data['Mental_Health_Score'] < 5:
                recommendations.append("🧠 Consider speaking with a counselor about mental health support")

            if input_data['Conflicts_Over_Social_Media'] > 3:
                recommendations.append("💬 Work on setting boundaries around social media with friends/family")

            if predicted_addiction > 6:
                recommendations.append("📵 Consider taking regular digital detox breaks")

            if not recommendations:
                recommendations.append("✨ You're doing great! Keep maintaining healthy habits.")

            for rec in recommendations:
                st.write(rec)

    else:
        # Default state when button hasn't been clicked
        st.info("👈 Fill in your information in the sidebar and click **Predict** to see results!")

        st.header("About This App")
        st.write("""
        This app uses machine learning to predict:

        1. **Academic Impact** - Whether social media usage affects your academic performance
        2. **Addiction Score** - A predicted severity score (1-10)
        3. **User Cluster** - Which group of users you're most similar to

        The predictions are based on 5 different machine learning models:
        - 🌳 Decision Tree
        - 📊 Naive Bayes
        - 🧠 Neural Network
        - 📈 Linear Regression
        - 👥 K-Means Clustering
        """)

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()