# ============================================================
# VISUALIZATION SCRIPT - SOCIAL MEDIA ADDICTION PROJECT
# ============================================================
# Run this AFTER training models in the notebook.
# Generates all charts for reports/presentations.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# SETUP
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs('visualizations', exist_ok=True)

print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
print("\n[1/10] Loading data and models...")

df = pd.read_csv(r"E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\archive\Students Social Media Addiction.csv")
df = df.drop(columns=['Student_ID'])

# Load models
dt_model = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\decision_tree_model.pkl')
nb_model = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\naive_bayes_model.pkl')
nn_model = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\neural_network_model.pkl')
lr_model = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\linear_regression_model.pkl')
kmeans_model = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\kmeans_model.pkl')
# Load scalers and feature info
scaler_class = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\scaler_class.pkl')
scaler_reg = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\scaler_reg.pkl')
scaler_kmeans = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\scaler_kmeans.pkl')
label_encoders = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\label_encoders.pkl')
feature_info = joblib.load(r'E:\DEPI\GIZ4_AIS2_S1\Wael Project Pianissimo\models\feature_info.pkl')

print("✓ All models loaded")

# ============================================================
# RECREATE FEATURES & SPLITS (needed for visualizations)
# ============================================================
print("\n[2/10] Recreating features and train/test splits...")

# Feature engineering (same as notebook)
try:
    import country_converter as coco
    fallback_map = {
        'UAE': 'Western Asia', 'UK': 'Northern Europe',
        'South Korea': 'Eastern Asia', 'Vietnam': 'South-eastern Asia',
        'Palestine': 'Western Asia'
    }
    df['Region'] = coco.convert(names=df['Country'].tolist(), to='UNregion', not_found=None)
    for idx in df[df['Region'].isna()].index:
        country = df.loc[idx, 'Country']
        df.loc[idx, 'Region'] = fallback_map.get(country, 'Other')
except ImportError:
    df['Region'] = 'Global'

df['Sleep_Deficit'] = np.maximum(0, 8 - df['Sleep_Hours_Per_Night'])
df['Usage_Sleep_Ratio'] = df['Avg_Daily_Usage_Hours'] / (df['Sleep_Hours_Per_Night'] + 0.1)

relationship_weights = {'Single': 1.0, 'In Relationship': 1.5, 'Complicated': 2.0}
df['Relationship_Strain'] = df.apply(
    lambda row: row['Conflicts_Over_Social_Media'] * relationship_weights.get(row['Relationship_Status'], 1.0),
    axis=1
)

# Addiction Risk Score
df['Usage_Norm'] = (df['Avg_Daily_Usage_Hours'] - df['Avg_Daily_Usage_Hours'].min()) / \
                   (df['Avg_Daily_Usage_Hours'].max() - df['Avg_Daily_Usage_Hours'].min())
df['Sleep_Deficit_Norm'] = (df['Sleep_Deficit'] - df['Sleep_Deficit'].min()) / \
                           (df['Sleep_Deficit'].max() - df['Sleep_Deficit'].min() + 0.001)
df['Conflict_Norm'] = (df['Conflicts_Over_Social_Media'] - df['Conflicts_Over_Social_Media'].min()) / \
                      (df['Conflicts_Over_Social_Media'].max() - df['Conflicts_Over_Social_Media'].min())
df['Mental_Health_Inv_Norm'] = 1 - ((df['Mental_Health_Score'] - df['Mental_Health_Score'].min()) / \
                                    (df['Mental_Health_Score'].max() - df['Mental_Health_Score'].min()))

df['Addiction_Risk_Score'] = (
    0.35 * df['Usage_Norm'] +
    0.25 * df['Sleep_Deficit_Norm'] +
    0.20 * df['Conflict_Norm'] +
    0.20 * df['Mental_Health_Inv_Norm']
)
df = df.drop(columns=['Usage_Norm', 'Sleep_Deficit_Norm', 'Conflict_Norm', 'Mental_Health_Inv_Norm'])

# Targets
y_class = (df['Affects_Academic_Performance'] == 'Yes').astype(int)
y_reg = df['Addicted_Score'].copy()

# Feature columns
feature_cols = [
    'Age', 'Gender', 'Academic_Level', 'Country',
    'Avg_Daily_Usage_Hours', 'Most_Used_Platform',
    'Sleep_Hours_Per_Night', 'Mental_Health_Score',
    'Relationship_Status', 'Conflicts_Over_Social_Media',
    'Region', 'Sleep_Deficit', 'Usage_Sleep_Ratio',
    'Relationship_Strain', 'Addiction_Risk_Score'
]
feature_cols_reg = feature_cols + ['Affects_Academic_Performance']

categorical_cols = ['Gender', 'Academic_Level', 'Country',
                    'Most_Used_Platform', 'Relationship_Status', 'Region']
numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                'Mental_Health_Score', 'Conflicts_Over_Social_Media',
                'Sleep_Deficit', 'Usage_Sleep_Ratio', 'Relationship_Strain',
                'Addiction_Risk_Score']

# Prepare X matrices
X_class_raw = df[feature_cols].copy()
X_reg_raw = df[feature_cols_reg].copy()

# One-hot encoding
X_class_onehot = pd.get_dummies(X_class_raw, columns=categorical_cols, drop_first=True)
X_reg_onehot = pd.get_dummies(X_reg_raw, columns=categorical_cols + ['Affects_Academic_Performance'], drop_first=True)

# Label encoding for Decision Tree
X_class_label = X_class_raw.copy()
for col in categorical_cols:
    le = label_encoders[col]
    X_class_label[col] = le.transform(X_class_label[col])

# Train/test split (same random state)
RANDOM_STATE = 42
TEST_SIZE = 0.3

X_train_oh, X_test_oh, y_train_class, y_test_class = train_test_split(
    X_class_onehot, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
)
X_train_lbl, X_test_lbl, _, _ = train_test_split(
    X_class_label, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_onehot, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Scale
X_train_oh[numeric_cols] = scaler_class.transform(X_train_oh[numeric_cols])
X_test_oh[numeric_cols] = scaler_class.transform(X_test_oh[numeric_cols])
X_train_reg[numeric_cols] = scaler_reg.transform(X_train_reg[numeric_cols])
X_test_reg[numeric_cols] = scaler_reg.transform(X_test_reg[numeric_cols])

# Get predictions
y_pred_dt = dt_model.predict(X_test_lbl)
y_prob_dt = dt_model.predict_proba(X_test_lbl)[:, 1]

y_pred_nb = nb_model.predict(X_test_oh)
y_prob_nb = nb_model.predict_proba(X_test_oh)[:, 1]

y_pred_nn = nn_model.predict(X_test_oh)
y_prob_nn = nn_model.predict_proba(X_test_oh)[:, 1]

y_pred_lr = lr_model.predict(X_test_reg)

# K-Means data
kmeans_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
               'Mental_Health_Score', 'Conflicts_Over_Social_Media',
               'Addicted_Score', 'Sleep_Deficit', 'Usage_Sleep_Ratio',
               'Relationship_Strain', 'Addiction_Risk_Score']
X_kmeans = df[kmeans_cols].copy()
X_kmeans_scaled = pd.DataFrame(
    scaler_kmeans.transform(X_kmeans),
    columns=kmeans_cols
)
clusters = kmeans_model.predict(X_kmeans_scaled)

print("✓ Data preparation complete")

# ============================================================
# VISUALIZATION 1: DATA EXPLORATION
# ============================================================
print("\n[3/10] Generating data exploration plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(df['Avg_Daily_Usage_Hours'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Distribution of Daily Usage Hours', fontsize=12)
axes[0, 0].set_xlabel('Hours')
axes[0, 0].set_ylabel('Count')

axes[0, 1].hist(df['Sleep_Hours_Per_Night'], bins=20, color='forestgreen', edgecolor='white')
axes[0, 1].set_title('Distribution of Sleep Hours', fontsize=12)
axes[0, 1].set_xlabel('Hours')
axes[0, 1].set_ylabel('Count')

axes[0, 2].hist(df['Addicted_Score'], bins=20, color='crimson', edgecolor='white')
axes[0, 2].set_title('Distribution of Addiction Score', fontsize=12)
axes[0, 2].set_xlabel('Score')
axes[0, 2].set_ylabel('Count')

axes[1, 0].hist(df['Mental_Health_Score'], bins=10, color='purple', edgecolor='white')
axes[1, 0].set_title('Distribution of Mental Health Score', fontsize=12)
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Count')

impact_counts = df['Affects_Academic_Performance'].value_counts()
axes[1, 1].pie(impact_counts, labels=impact_counts.index, autopct='%1.1f%%',
               colors=['#ff9999', '#90EE90'], explode=(0.02, 0.02))
axes[1, 1].set_title('Affects Academic Performance', fontsize=12)

platform_counts = df['Most_Used_Platform'].value_counts().head(8)
bars = axes[1, 2].barh(platform_counts.index, platform_counts.values, color='teal')
axes[1, 2].set_title('Top 8 Platforms Used', fontsize=12)
axes[1, 2].set_xlabel('Count')
axes[1, 2].bar_label(bars, padding=3)

plt.suptitle('Data Exploration', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/01_data_exploration.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_data_exploration.png")

# ============================================================
# VISUALIZATION 2: CORRELATION MATRIX
# ============================================================
print("\n[4/10] Generating correlation matrix...")

numeric_for_corr = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score',
                    'Sleep_Deficit', 'Usage_Sleep_Ratio', 'Relationship_Strain', 'Addiction_Risk_Score']

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[numeric_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.2f', linewidths=0.5, ax=ax, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix (Including Engineered Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/02_correlation_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_correlation_matrix.png")

# ============================================================
# VISUALIZATION 3: FEATURE DISTRIBUTIONS BY TARGET
# ============================================================
print("\n[5/10] Generating feature distributions by target...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Usage by Academic Impact
sns.boxplot(data=df, x='Affects_Academic_Performance', y='Avg_Daily_Usage_Hours',
            ax=axes[0, 0], palette=['lightgreen', 'lightcoral'])
axes[0, 0].set_title('Daily Usage by Academic Impact')
axes[0, 0].set_xlabel('Affects Academic Performance')
axes[0, 0].set_ylabel('Hours')

# Sleep by Academic Impact
sns.boxplot(data=df, x='Affects_Academic_Performance', y='Sleep_Hours_Per_Night',
            ax=axes[0, 1], palette=['lightgreen', 'lightcoral'])
axes[0, 1].set_title('Sleep Hours by Academic Impact')
axes[0, 1].set_xlabel('Affects Academic Performance')
axes[0, 1].set_ylabel('Hours')

# Usage vs Addiction colored by impact
scatter = axes[1, 0].scatter(
    df['Avg_Daily_Usage_Hours'],
    df['Addicted_Score'],
    c=y_class,
    cmap='RdYlGn_r',
    alpha=0.6,
    s=40
)
axes[1, 0].set_xlabel('Daily Usage (hours)')
axes[1, 0].set_ylabel('Addiction Score')
axes[1, 0].set_title('Usage vs Addiction (colored by Academic Impact)')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['No Impact', 'Has Impact'])

# Addiction Risk Score distribution
sns.histplot(data=df, x='Addiction_Risk_Score', hue='Affects_Academic_Performance',
             ax=axes[1, 1], kde=True, palette=['green', 'red'], alpha=0.5)
axes[1, 1].set_title('Addiction Risk Score Distribution')
axes[1, 1].set_xlabel('Risk Score')

plt.suptitle('Feature Analysis by Target Variable', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/03_feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_feature_distributions.png")

# ============================================================
# VISUALIZATION 4: DECISION TREE STRUCTURE
# ============================================================
print("\n[6/10] Generating decision tree visualization...")

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    dt_model,
    max_depth=3,
    filled=True,
    rounded=True,
    feature_names=X_train_lbl.columns.tolist(),
    class_names=['No Impact', 'Has Impact'],
    fontsize=9,
    ax=ax
)
ax.set_title('Decision Tree Structure (Max Depth = 3 for Visibility)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/04_decision_tree.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_decision_tree.png")

# ============================================================
# VISUALIZATION 5: FEATURE IMPORTANCE
# ============================================================
print("\n[7/10] Generating feature importance chart...")

importance_df = pd.DataFrame({
    'Feature': X_train_lbl.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance (Decision Tree)', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig('visualizations/05_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_feature_importance.png")

# ============================================================
# VISUALIZATION 6: CONFUSION MATRICES
# ============================================================
print("\n[8/10] Generating confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models_data = [
    ('Decision Tree', y_pred_dt),
    ('Naive Bayes', y_pred_nb),
    ('Neural Network', y_pred_nn)
]

for ax, (name, y_pred) in zip(axes, models_data):
    cm = confusion_matrix(y_test_class, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Impact', 'Has Impact'],
                yticklabels=['No Impact', 'Has Impact'],
                annot_kws={'size': 14})
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices - Classification Models', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('visualizations/06_confusion_matrices.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_confusion_matrices.png")

# ============================================================
# VISUALIZATION 7: ROC CURVES
# ============================================================
print("\n[9/10] Generating ROC curves...")

fig, ax = plt.subplots(figsize=(8, 6))

roc_data = [
    ('Decision Tree', y_prob_dt, '#3498db'),
    ('Naive Bayes', y_prob_nb, '#2ecc71'),
    ('Neural Network', y_prob_nn, '#e74c3c')
]

for name, y_prob, color in roc_data:
    fpr, tpr, _ = roc_curve(y_test_class, y_prob)
    auc_score = roc_auc_score(y_test_class, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC = {auc_score:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.500)')
ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Classification Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/07_roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_roc_curves.png")

# ============================================================
# VISUALIZATION 8: MODEL COMPARISON
# ============================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_data = {
    'Model': ['Decision Tree', 'Naive Bayes', 'Neural Network'],
    'Accuracy': [
        accuracy_score(y_test_class, y_pred_dt),
        accuracy_score(y_test_class, y_pred_nb),
        accuracy_score(y_test_class, y_pred_nn)
    ],
    'Precision': [
        precision_score(y_test_class, y_pred_dt),
        precision_score(y_test_class, y_pred_nb),
        precision_score(y_test_class, y_pred_nn)
    ],
    'Recall': [
        recall_score(y_test_class, y_pred_dt),
        recall_score(y_test_class, y_pred_nb),
        recall_score(y_test_class, y_pred_nn)
    ],
    'F1-Score': [
        f1_score(y_test_class, y_pred_dt),
        f1_score(y_test_class, y_pred_nb),
        f1_score(y_test_class, y_pred_nn)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test_class, y_prob_dt),
        roc_auc_score(y_test_class, y_prob_nb),
        roc_auc_score(y_test_class, y_prob_nn)
    ]
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(3)
width = 0.15
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']):
    bars = ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=colors[i])
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Models - Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics_df['Model'], fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.5, 1.1])
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/08_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_model_comparison.png")

# ============================================================
# VISUALIZATION 9: LINEAR REGRESSION ANALYSIS
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr_r2 = r2_score(y_test_reg, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
residuals = y_test_reg - y_pred_lr

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Actual vs Predicted
axes[0].scatter(y_test_reg, y_pred_lr, alpha=0.5, s=30, color='steelblue')
axes[0].plot([y_test_reg.min(), y_test_reg.max()],
             [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Addiction Score', fontsize=11)
axes[0].set_ylabel('Predicted Addiction Score', fontsize=11)
axes[0].set_title(f'Actual vs Predicted (R² = {lr_r2:.3f})', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual plot
axes[1].scatter(y_pred_lr, residuals, alpha=0.5, s=30, color='forestgreen')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Value', fontsize=11)
axes[1].set_ylabel('Residual', fontsize=11)
axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Residual histogram
axes[2].hist(residuals, bins=30, color='purple', edgecolor='white', alpha=0.7)
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[2].axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {residuals.mean():.3f}')
axes[2].set_xlabel('Residual Value', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[2].legend()

plt.suptitle('Linear Regression Analysis (Predicting Addiction Score)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('visualizations/09_linear_regression.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_linear_regression.png")

# ============================================================
# VISUALIZATION 10: K-MEANS CLUSTERING
# ============================================================
print("\n[10/10] Generating K-Means clustering visualizations...")

# Elbow and Silhouette analysis
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_kmeans_scaled)
    inertias.append(km_temp.inertia_)
    silhouettes.append(silhouette_score(X_kmeans_scaled, km_temp.labels_))

optimal_k = kmeans_model.n_clusters

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Elbow plot
axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen K={optimal_k}')
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 0].set_ylabel('Inertia', fontsize=11)
axes[0, 0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(list(K_range))
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Silhouette plot
axes[0, 1].plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Chosen K={optimal_k}')
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
axes[0, 1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(list(K_range))
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cluster scatter: Usage vs Sleep
scatter1 = axes[1, 0].scatter(
    df['Avg_Daily_Usage_Hours'],
    df['Sleep_Hours_Per_Night'],
    c=clusters,
    cmap='viridis',
    alpha=0.6,
    s=50,
    edgecolors='white',
    linewidth=0.5
)
axes[1, 0].set_xlabel('Daily Usage (hours)', fontsize=11)
axes[1, 0].set_ylabel('Sleep (hours)', fontsize=11)
axes[1, 0].set_title('Clusters: Usage vs Sleep', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[1, 0], label='Cluster')

# Cluster scatter: Usage vs Addiction Score
scatter2 = axes[1, 1].scatter(
    df['Avg_Daily_Usage_Hours'],
    df['Addicted_Score'],
    c=clusters,
    cmap='viridis',
    alpha=0.6,
    s=50,
    edgecolors='white',
    linewidth=0.5
)
axes[1, 1].set_xlabel('Daily Usage (hours)', fontsize=11)
axes[1, 1].set_ylabel('Addiction Score', fontsize=11)
axes[1, 1].set_title('Clusters: Usage vs Addiction Score', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=axes[1, 1], label='Cluster')

plt.suptitle(f'K-Means Clustering Analysis (K={optimal_k})', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/10_kmeans_clustering.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 10_kmeans_clustering.png")

# ============================================================
# BONUS: CLUSTER PROFILES TABLE
# ============================================================
print("\n[BONUS] Generating cluster profiles...")

df_clustered = df.copy()
df_clustered['Cluster'] = clusters

cluster_profiles = df_clustered.groupby('Cluster').agg({
    'Avg_Daily_Usage_Hours': 'mean',
    'Sleep_Hours_Per_Night': 'mean',
    'Addicted_Score': 'mean',
    'Mental_Health_Score': 'mean',
    'Conflicts_Over_Social_Media': 'mean',
    'Addiction_Risk_Score': 'mean'
}).round(2)

# Save cluster profiles to CSV
cluster_profiles.to_csv('visualizations/cluster_profiles.csv')
print("  ✓ Saved: cluster_profiles.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nFiles saved to 'visualizations/' folder:")
print("  1. 01_data_exploration.png")
print("  2. 02_correlation_matrix.png")
print("  3. 03_feature_distributions.png")
print("  4. 04_decision_tree.png")
print("  5. 05_feature_importance.png")
print("  6. 06_confusion_matrices.png")
print("  7. 07_roc_curves.png")
print("  8. 08_model_comparison.png")
print("  9. 09_linear_regression.png")
print("  10. 10_kmeans_clustering.png")
print("  11. cluster_profiles.csv")
print("\n✓ All visualizations ready for your report!")