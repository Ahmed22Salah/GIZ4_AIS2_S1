"""
Social Media Addiction Predictor - Streamlit Web Application
============================================================
This app uses machine learning models to predict:
1. Whether social media affects academic performance (Classification)
2. Addiction severity score (Regression)
3. User behavior cluster (Clustering)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS (CACHED)
# ============================================================
@st.cache_resource
def load_models():
    """Load all saved models and artifacts."""
    try:
        models = {
            'decision_tree': joblib.load('models/decision_tree_model.pkl'),
            'naive_bayes': joblib.load('models/naive_bayes_model.pkl'),
            'neural_network': joblib.load('models/neural_network_model.pkl'),
            'linear_regression': joblib.load('models/linear_regression_model.pkl'),
            'kmeans': joblib.load('models/kmeans_model.pkl'),
            'label_encoders': joblib.load('models/label_encoders.pkl')
        }
        
        scalers = {
            'class': joblib.load('models/scaler_class.pkl'),
            'reg': joblib.load('models/scaler_reg.pkl'),
            'kmeans': joblib.load('models/scaler_kmeans.pkl')
        }
        
        feature_info = joblib.load('models/feature_info.pkl')
        
        return models, scalers, feature_info, None
    
    except FileNotFoundError as e:
        return None, None, None, str(e)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_engineered_features(input_data):
    """Create engineered features from raw input."""
    data = input_data.copy()
    
    # Sleep Deficit (hours below 8)
    data['Sleep_Deficit'] = max(0, 8 - data['Sleep_Hours_Per_Night'])
    
    # Usage-Sleep Ratio
    data['Usage_Sleep_Ratio'] = data['Avg_Daily_Usage_Hours'] / (data['Sleep_Hours_Per_Night'] + 0.1)
    
    # Relationship Strain Index
    relationship_weights = {'Single': 1.0, 'In Relationship': 1.5, 'Complicated': 2.0}
    weight = relationship_weights.get(data['Relationship_Status'], 1.0)
    data['Relationship_Strain'] = data['Conflicts_Over_Social_Media'] * weight
    
    # Addiction Risk Score (simplified version)
    # Normalize components to 0-1 range (using typical ranges)
    usage_norm = min(data['Avg_Daily_Usage_Hours'] / 15.0, 1.0)
    sleep_deficit_norm = min(data['Sleep_Deficit'] / 6.0, 1.0)
    conflict_norm = (data['Conflicts_Over_Social_Media'] - 1) / 9.0
    mental_health_inv_norm = (10 - data['Mental_Health_Score']) / 9.0
    
    data['Addiction_Risk_Score'] = (
        0.35 * usage_norm +
        0.25 * sleep_deficit_norm +
        0.20 * conflict_norm +
        0.20 * mental_health_inv_norm
    )
    
    return data


def prepare_label_encoded_input(input_data, label_encoders):
    """Prepare input for Decision Tree (label encoded)."""
    
    feature_cols = [
        'Age', 'Gender', 'Academic_Level', 'Country',
        'Avg_Daily_Usage_Hours', 'Most_Used_Platform',
        'Sleep_Hours_Per_Night', 'Mental_Health_Score',
        'Relationship_Status', 'Conflicts_Over_Social_Media',
        'Region', 'Sleep_Deficit', 'Usage_Sleep_Ratio',
        'Relationship_Strain', 'Addiction_Risk_Score'
    ]
    
    # Create DataFrame
    df_input = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_cols}])
    
    # Apply label encoding to categorical columns
    categorical_cols = ['Gender', 'Academic_Level', 'Country', 
                        'Most_Used_Platform', 'Relationship_Status', 'Region']
    
    for col in categorical_cols:
        if col in label_encoders and col in df_input.columns:
            le = label_encoders[col]
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError:
                # If value wasn't seen during training, use 0
                df_input[col] = 0
    
    return df_input[feature_cols]


def prepare_onehot_encoded_input(input_data, feature_info, scaler):
    """Prepare input for Naive Bayes and Neural Network (one-hot encoded)."""
    
    expected_cols = feature_info['X_class_columns']
    
    # Create DataFrame with all expected columns, initialized to 0
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)
    
    # Numeric columns
    numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Mental_Health_Score', 'Conflicts_Over_Social_Media',
                    'Sleep_Deficit', 'Usage_Sleep_Ratio', 'Relationship_Strain',
                    'Addiction_Risk_Score']
    
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
        'Region': input_data.get('Region', 'Northern America')
    }
    
    for cat_col, value in categorical_mappings.items():
        possible_col_names = [
            f"{cat_col}_{value}",
            f"{cat_col}_{value}".replace(" ", "_"),
            f"{cat_col}_{value}".replace("-", "_"),
        ]
        
        for col_name in possible_col_names:
            if col_name in df_input.columns:
                df_input.loc[0, col_name] = 1
                break
    
    # Scale numeric features
    numeric_cols_in_df = [col for col in numeric_cols if col in df_input.columns]
    
    if len(numeric_cols_in_df) > 0:
        try:
            df_input[numeric_cols_in_df] = scaler.transform(df_input[numeric_cols_in_df])
        except Exception:
            pass  # If scaling fails, use unscaled values
    
    return df_input


def prepare_regression_input(input_data, feature_info, scaler):
    """Prepare input for regression model."""
    
    expected_cols = feature_info['X_reg_columns']
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)
    
    numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Mental_Health_Score', 'Conflicts_Over_Social_Media',
                    'Sleep_Deficit', 'Usage_Sleep_Ratio', 'Relationship_Strain',
                    'Addiction_Risk_Score']
    
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
        'Affects_Academic_Performance': 'No'  # Default value
    }
    
    for cat_col, value in categorical_mappings.items():
        possible_col_names = [
            f"{cat_col}_{value}",
            f"{cat_col}_{value}".replace(" ", "_"),
            f"{cat_col}_{value}".replace("-", "_"),
        ]
        
        for col_name in possible_col_names:
            if col_name in df_input.columns:
                df_input.loc[0, col_name] = 1
                break
    
    # Scale numeric features
    numeric_cols_in_df = [col for col in numeric_cols if col in df_input.columns]
    
    if len(numeric_cols_in_df) > 0:
        try:
            df_input[numeric_cols_in_df] = scaler.transform(df_input[numeric_cols_in_df])
        except Exception:
            pass
    
    return df_input


def prepare_kmeans_input(input_data, feature_info, scaler):
    """Prepare input for K-Means clustering."""
    
    kmeans_cols = feature_info['kmeans_columns']
    
    # Create input with available features
    kmeans_data = {}
    for col in kmeans_cols:
        if col in input_data:
            kmeans_data[col] = input_data[col]
        elif col == 'Addicted_Score':
            kmeans_data[col] = 5  # Placeholder
        else:
            kmeans_data[col] = 0
    
    df_input = pd.DataFrame([kmeans_data])
    df_input = df_input[kmeans_cols]  # Ensure correct order
    
    # Scale
    try:
        df_scaled = scaler.transform(df_input)
        return df_scaled
    except Exception:
        return df_input.values


def create_gauge_chart(value, title, max_val=10):
    """Create a gauge chart for displaying scores."""
    
    # Determine color based on value
    if value <= max_val * 0.3:
        color = "green"
    elif value <= max_val * 0.6:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_val * 0.3], 'color': 'lightgreen'},
                {'range': [max_val * 0.3, max_val * 0.6], 'color': 'lightyellow'},
                {'range': [max_val * 0.6, max_val], 'color': 'lightcoral'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_probability_chart(probabilities):
    """Create a bar chart showing model probabilities."""
    
    fig = go.Figure()
    
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    for i, (name, prob) in enumerate(probabilities.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[prob * 100],
            marker_color=colors[i % len(colors)],
            text=[f'{prob*100:.1f}%'],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Model Predictions (Risk Probability %)',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application function."""
    
    # ==================== HEADER ====================
    st.title("📱 Social Media Addiction Predictor")
    st.markdown("""
    This application uses **machine learning** to predict how social media usage 
    affects academic performance and overall wellbeing.
    """)
    
    # ==================== LOAD MODELS ====================
    models, scalers, feature_info, error = load_models()
    
    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("Please make sure you've run the notebook first to generate the model files.")
        st.stop()
    
    # ==================== SIDEBAR INPUTS ====================
    st.sidebar.header("📝 Enter Your Information")
    st.sidebar.markdown("---")
    
    # Personal Information
    st.sidebar.subheader("👤 Personal Info")
    
    age = st.sidebar.slider(
        "Age",
        min_value=14,
        max_value=35,
        value=20,
        help="Your current age"
    )
    
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Male", "Female"],
        help="Select your gender"
    )
    
    academic_level = st.sidebar.selectbox(
        "Academic Level",
        options=["High School", "Undergraduate", "Postgraduate"],
        help="Your current education level"
    )
    
    country = st.sidebar.selectbox(
        "Country",
        options=[
            "USA", "UK", "India", "Canada", "Australia", 
            "Germany", "France", "Brazil", "Japan", "South Korea", 
            "Mexico", "UAE", "China", "Indonesia", "Nigeria", "Other"
        ],
        help="Your country of residence"
    )
    
    relationship_status = st.sidebar.selectbox(
        "Relationship Status",
        options=["Single", "In Relationship", "Complicated"],
        help="Your current relationship status"
    )
    
    st.sidebar.markdown("---")
    
    # Social Media Usage
    st.sidebar.subheader("📲 Social Media Usage")
    
    most_used_platform = st.sidebar.selectbox(
        "Most Used Platform",
        options=[
            "Instagram", "TikTok", "Facebook", "Twitter/X", 
            "YouTube", "Snapchat", "LinkedIn", "WhatsApp", 
            "Reddit", "Pinterest", "Other"
        ],
        help="The social media platform you use most"
    )
    
    avg_daily_usage = st.sidebar.slider(
        "Average Daily Usage (hours)",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.5,
        help="How many hours per day do you spend on social media?"
    )
    
    conflicts = st.sidebar.slider(
        "Conflicts Over Social Media",
        min_value=1,
        max_value=10,
        value=3,
        help="How often do you have conflicts related to social media? (1=Never, 10=Very Often)"
    )
    
    st.sidebar.markdown("---")
    
    # Health & Wellbeing
    st.sidebar.subheader("🏥 Health & Wellbeing")
    
    sleep_hours = st.sidebar.slider(
        "Sleep Hours Per Night",
        min_value=2.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        help="Average hours of sleep you get per night"
    )
    
    mental_health_score = st.sidebar.slider(
        "Mental Health Score",
        min_value=1,
        max_value=10,
        value=6,
        help="Rate your mental health (1=Very Poor, 10=Excellent)"
    )
    
    st.sidebar.markdown("---")
    
    # ==================== COLLECT INPUT DATA ====================
    # Map country to region
    region_mapping = {
        'USA': 'Northern America',
        'Canada': 'Northern America',
        'UK': 'Northern Europe',
        'Germany': 'Western Europe',
        'France': 'Western Europe',
        'India': 'Southern Asia',
        'Japan': 'Eastern Asia',
        'South Korea': 'Eastern Asia',
        'China': 'Eastern Asia',
        'Australia': 'Australia and New Zealand',
        'Brazil': 'South America',
        'Mexico': 'Central America',
        'UAE': 'Western Asia',
        'Indonesia': 'South-eastern Asia',
        'Nigeria': 'Western Africa',
        'Other': 'Other'
    }
    
    input_data = {
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Country': country,
        'Most_Used_Platform': most_used_platform,
        'Avg_Daily_Usage_Hours': avg_daily_usage,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Mental_Health_Score': mental_health_score,
        'Relationship_Status': relationship_status,
        'Conflicts_Over_Social_Media': conflicts,
        'Region': region_mapping.get(country, 'Other')
    }
    
    # ==================== PREDICT BUTTON ====================
    predict_button = st.sidebar.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True
    )
    
    # ==================== PREDICTION LOGIC ====================
    if predict_button:
        
        with st.spinner("Analyzing your data..."):
            
            # Create engineered features
            input_data = create_engineered_features(input_data)
            
            # Prepare inputs for different model types
            X_label = prepare_label_encoded_input(input_data, models['label_encoders'])
            X_onehot = prepare_onehot_encoded_input(input_data, feature_info, scalers['class'])
            X_reg = prepare_regression_input(input_data, feature_info, scalers['reg'])
            X_kmeans = prepare_kmeans_input(input_data, feature_info, scalers['kmeans'])
            
            # ==================== CLASSIFICATION PREDICTIONS ====================
            predictions = {}
            probabilities = {}
            
            # Decision Tree (uses label-encoded data)
            try:
                dt_pred = models['decision_tree'].predict(X_label)[0]
                dt_prob = models['decision_tree'].predict_proba(X_label)[0][1]
                predictions['Decision Tree'] = dt_pred
                probabilities['Decision Tree'] = dt_prob
            except Exception as e:
                st.warning(f"Decision Tree prediction failed: {e}")
                predictions['Decision Tree'] = 0
                probabilities['Decision Tree'] = 0.5
            
            # Naive Bayes (uses one-hot encoded data)
            try:
                nb_pred = models['naive_bayes'].predict(X_onehot)[0]
                nb_prob = models['naive_bayes'].predict_proba(X_onehot)[0][1]
                predictions['Naive Bayes'] = nb_pred
                probabilities['Naive Bayes'] = nb_prob
            except Exception as e:
                st.warning(f"Naive Bayes prediction failed: {e}")
                predictions['Naive Bayes'] = 0
                probabilities['Naive Bayes'] = 0.5
            
            # Neural Network (uses one-hot encoded data)
            try:
                nn_pred = models['neural_network'].predict(X_onehot)[0]
                nn_prob = models['neural_network'].predict_proba(X_onehot)[0][1]
                predictions['Neural Network'] = nn_pred
                probabilities['Neural Network'] = nn_prob
            except Exception as e:
                st.warning(f"Neural Network prediction failed: {e}")
                predictions['Neural Network'] = 0
                probabilities['Neural Network'] = 0.5
            
            # Ensemble prediction (majority vote)
            ensemble_pred = 1 if sum(predictions.values()) >= 2 else 0
            avg_prob = sum(probabilities.values()) / len(probabilities)
            
            # ==================== REGRESSION PREDICTION ====================
            try:
                predicted_addiction = models['linear_regression'].predict(X_reg)[0]
                predicted_addiction = np.clip(predicted_addiction, 1, 10)
            except Exception as e:
                st.warning(f"Regression prediction failed: {e}")
                predicted_addiction = 5.0
            
            # ==================== CLUSTERING PREDICTION ====================
            try:
                cluster = models['kmeans'].predict(X_kmeans)[0]
            except Exception as e:
                st.warning(f"Clustering prediction failed: {e}")
                cluster = 0
        
        # ==================== DISPLAY RESULTS ====================
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Academic Impact
        with col1:
            st.subheader("📚 Academic Impact")
            
            if ensemble_pred == 1:
                st.error("⚠️ **YES** - Social media likely affects your academics")
            else:
                st.success("✅ **NO** - Social media unlikely to affect your academics")
            
            st.metric(
                label="Risk Probability",
                value=f"{avg_prob * 100:.1f}%",
                delta=None
            )
            
            # Model breakdown
            with st.expander("📊 Model Breakdown"):
                for name, prob in probabilities.items():
                    pred_text = "⚠️ Yes" if predictions[name] == 1 else "✅ No"
                    st.write(f"**{name}:** {pred_text} ({prob*100:.1f}% risk)")
        
        # Column 2: Addiction Score
        with col2:
            st.subheader("📈 Addiction Score")
            
            fig_gauge = create_gauge_chart(
                predicted_addiction, 
                "Predicted Score",
                max_val=10
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if predicted_addiction <= 3:
                st.success("🟢 Low addiction level")
            elif predicted_addiction <= 6:
                st.warning("🟡 Moderate addiction level")
            else:
                st.error("🔴 High addiction level")
        
        # Column 3: User Cluster
        with col3:
            st.subheader("👥 User Cluster")
            
            st.metric(
                label="Your Group",
                value=f"Cluster {cluster}",
                delta=None
            )
            
            # Cluster descriptions
            cluster_descriptions = {
                0: "🟢 **Healthy Users**: Balanced social media habits with good sleep and mental health.",
                1: "🟡 **Moderate Users**: Average usage with some areas for improvement.",
                2: "🟠 **At-Risk Users**: Higher usage patterns that may need attention.",
                3: "🔴 **Heavy Users**: Significant usage that may be impacting wellbeing.",
                4: "⚫ **Critical Users**: Very high usage with potential negative effects."
            }
            
            desc = cluster_descriptions.get(cluster, "User profile identified based on your responses.")
            st.markdown(desc)
        
        # ==================== PROBABILITY CHART ====================
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Comparison")
            fig_probs = create_probability_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)
        
        with col2:
            st.subheader("📋 Your Input Summary")
            
            summary_data = {
                "Metric": [
                    "Daily Usage",
                    "Sleep Hours",
                    "Sleep Deficit",
                    "Mental Health",
                    "Conflicts",
                    "Risk Score"
                ],
                "Value": [
                    f"{input_data['Avg_Daily_Usage_Hours']:.1f} hours",
                    f"{input_data['Sleep_Hours_Per_Night']:.1f} hours",
                    f"{input_data['Sleep_Deficit']:.1f} hours",
                    f"{input_data['Mental_Health_Score']}/10",
                    f"{input_data['Conflicts_Over_Social_Media']}/10",
                    f"{input_data['Addiction_Risk_Score']:.2f}"
                ]
            }
            
            st.table(pd.DataFrame(summary_data))
        
        # ==================== RECOMMENDATIONS ====================
        st.markdown("---")
        st.header("💡 Personalized Recommendations")
        
        recommendations = []
        
        if input_data['Avg_Daily_Usage_Hours'] > 5:
            recommendations.append({
                "icon": "🕐",
                "title": "Reduce Screen Time",
                "text": "Consider reducing daily social media usage to under 5 hours. Try setting app timers or designated 'phone-free' periods."
            })
        
        if input_data['Sleep_Hours_Per_Night'] < 7:
            recommendations.append({
                "icon": "😴",
                "title": "Improve Sleep",
                "text": "Aim for 7-8 hours of sleep. Avoid screens 1 hour before bed and establish a consistent sleep schedule."
            })
        
        if input_data['Mental_Health_Score'] < 5:
            recommendations.append({
                "icon": "🧠",
                "title": "Mental Health Support",
                "text": "Consider speaking with a counselor or mental health professional. Many schools offer free counseling services."
            })
        
        if input_data['Conflicts_Over_Social_Media'] > 5:
            recommendations.append({
                "icon": "💬",
                "title": "Communication",
                "text": "Work on setting healthy boundaries around social media use with friends and family."
            })
        
        if predicted_addiction > 6:
            recommendations.append({
                "icon": "📵",
                "title": "Digital Detox",
                "text": "Consider taking regular breaks from social media. Try a 24-hour detox on weekends."
            })
        
        if input_data['Usage_Sleep_Ratio'] > 1:
            recommendations.append({
                "icon": "⚖️",
                "title": "Balance Usage and Rest",
                "text": "You're spending more time on social media than sleeping. Try to reverse this ratio for better health."
            })
        
        if not recommendations:
            recommendations.append({
                "icon": "✨",
                "title": "Keep It Up!",
                "text": "You're doing great! Keep maintaining your healthy social media habits."
            })
        
        # Display recommendations in columns
        cols = st.columns(min(len(recommendations), 3))
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;">
                    <h4>{rec['icon']} {rec['title']}</h4>
                    <p>{rec['text']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ==================== DISCLAIMER ====================
        st.markdown("---")
        st.caption("""
        **Disclaimer:** This tool provides predictions based on machine learning models trained on survey data. 
        Results are for informational purposes only and should not be considered medical or professional advice. 
        If you're struggling with social media addiction or mental health issues, please consult a healthcare professional.
        """)
    
    else:
        # ==================== DEFAULT STATE ====================
        st.info("👈 Fill in your information in the sidebar and click **Predict** to see results!")
        
        st.markdown("---")
        
        # About section
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📖 About This App")
            st.markdown("""
            This application uses **5 different machine learning models** to analyze 
            social media usage patterns and predict potential impacts:
            
            **Classification Models** (Yes/No predictions):
            - 🌳 **Decision Tree** - Rule-based classification
            - 📊 **Naive Bayes** - Probabilistic classification
            - 🧠 **Neural Network** - Deep learning classification
            
            **Regression Model** (Score prediction):
            - 📈 **Linear Regression** - Predicts addiction severity score
            
            **Clustering Model** (User grouping):
            - 👥 **K-Means** - Groups similar users together
            """)
        
        with col2:
            st.header("🔬 How It Works")
            st.markdown("""
            1. **Enter your information** in the sidebar
            2. **Click Predict** to run the analysis
            3. **View results** including:
               - Whether social media affects your academics
               - Your predicted addiction score (1-10)
               - Which user cluster you belong to
               - Personalized recommendations
            
            The models were trained on survey data from students 
            and consider factors like:
            - Daily usage hours
            - Sleep patterns
            - Mental health scores
            - Relationship conflicts
            """)
        
        # Sample predictions
        st.markdown("---")
        st.header("📊 Sample Analysis")
        
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            st.markdown("""
            **🟢 Low Risk Profile:**
            - 2 hours daily usage
            - 8 hours sleep
            - Good mental health (8/10)
            - Few conflicts (2/10)
            """)
        
        with sample_col2:
            st.markdown("""
            **🟡 Moderate Risk Profile:**
            - 5 hours daily usage
            - 6 hours sleep
            - Average mental health (5/10)
            - Some conflicts (5/10)
            """)
        
        with sample_col3:
            st.markdown("""
            **🔴 High Risk Profile:**
            - 8+ hours daily usage
            - 4 hours sleep
            - Poor mental health (3/10)
            - Frequent conflicts (8/10)
            """)


# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    main()