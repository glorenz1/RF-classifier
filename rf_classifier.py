import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Load & label
df = pd.read_csv('data.csv')

def classify_chip(row):
    if row['open'] > 600:
        return 'open bumps'
    elif row['dead'] > 600 or row['fiterror'] > 1000:
        return 'tuning problems'
    elif row['masked'] > 200:
        return 'readout problems'
    else:
        return 'working good'

df['label'] = df.apply(classify_chip, axis=1)
print("\n====== Classification ======")
print(df['label'].value_counts())

target_names = ['working good', 'tuning problems', 'open bumps', 'readout problems']
target_names = [t for t in target_names if t in df['label'].unique()]

le = LabelEncoder()
le.classes_ = np.array(target_names)
y = le.transform(df['label'])

# Define features, split & scale
features = [
    'I digital [mA]',
    'analog [mA]',
    'thr_entries',
    'thr_mean',
    'thr_std',
    'thr_rms',
    'noise_mean',
    'noise_std',
    'noise_rms',
    'open',
    'dead',
    'fiterror',
    'masked',
    'dead from start',
]

X = df[features].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Initial random forest
print("\n====== Baseline Feature Accuracy ======")
clf_base = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf_base.fit(X_train_scaled, y_train)
y_pred_base = clf_base.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_base) * 100:.2f}%")

# Correlation matrix 
corr_base = df[features].corr()
mask = np.triu(np.ones_like(corr_base, dtype=bool), k=1)

plt.figure(figsize=(14, 12))
sns.heatmap(corr_base, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, annot_kws={'size': 10}, linewidths=0.5, square=True)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Confusion matrix
conf_base = confusion_matrix(y_test, y_pred_base, labels=range(len(target_names)))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_base, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png')
plt.close()

# Feature importance
imp_features = clf_base.feature_importances_.argsort()

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in imp_features], clf_base.feature_importances_[imp_features])
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('baseline_feature_importance.png')
plt.close()

# RFECV feature reduction
print("\n====== RFECV ======")

rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    step=1,
    cv=StratifiedKFold(n_splits=3), # Change n_splits based on number of rare cases
    scoring='accuracy',
)
rfecv.fit(X_train_scaled, y_train)

selected_features   = [features[i] for i in range(len(features)) if rfecv.support_[i]]
eliminated_features = [features[i] for i in range(len(features)) if not rfecv.support_[i]]
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features:      {selected_features}")
print(f"Eliminated features:    {eliminated_features}")

# RFECV accuracy curve
cv_mean = rfecv.cv_results_['mean_test_score']
cv_std  = rfecv.cv_results_['std_test_score']
n_feat_range = range(1, len(cv_mean) + 1)

plt.figure(figsize=(10, 5))
plt.plot(n_feat_range, cv_mean, marker='o', label='Mean CV accuracy')
plt.fill_between(n_feat_range, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, label='±1 std')
plt.axvline(rfecv.n_features_, color='red', linestyle='--', label=f'Optimal: {rfecv.n_features_} features')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('rfecv_accuracy.png')
plt.close()

X_train_rfecv = rfecv.transform(X_train_scaled)
X_test_rfecv  = rfecv.transform(X_test_scaled)

# Random Forest on selected features
print("\n====== Reduced Feature Accuracy ======")
clf_rfecv = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf_rfecv.fit(X_train_rfecv, y_train)
y_pred_rfecv = clf_rfecv.predict(X_test_rfecv)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rfecv) * 100:.2f}%")

# Confusion matrix after RFECV
conf_rfecv = confusion_matrix(y_test, y_pred_rfecv, labels=range(len(target_names)))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_rfecv, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('rfecv_confusion_matrix.png')
plt.close()

# Feature importance after RFECV
imp_features_rfecv = clf_rfecv.feature_importances_.argsort()

plt.figure(figsize=(10, 6))
plt.barh([selected_features[i] for i in imp_features_rfecv], clf_rfecv.feature_importances_[imp_features_rfecv])
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('rfecv_feature_importance.png')
plt.close()

print("\n====== Summary ======")
print(f"Baseline accuracy ({len(features)} features): {accuracy_score(y_test, y_pred_base) * 100:.2f}%")
print(f"Reduction accuracy ({rfecv.n_features_} features): {accuracy_score(y_test, y_pred_rfecv) * 100:.2f}%")
