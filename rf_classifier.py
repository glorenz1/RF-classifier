import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')

def classify_chip(row):
    if row['open'] > 600:
        return 'open bumps'
    elif row['dead'] > 600 or row['fiterror'] > 1000:
        return 'tuning problems'
    else:
        return 'working good'

df['label'] = df.apply(classify_chip, axis=1)

print("Classification:")
print(df['label'].value_counts())

target_names = ['working good', 'tuning problems', 'open bumps', 'readout problems']
target_names = [t for t in target_names if t in df['label'].unique()]

le = LabelEncoder()
le.classes_ = np.array(target_names)
y = le.transform(df['label'])

feature_names = [
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

X = df[feature_names].values

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# train random forest
clf = RandomForestClassifier(n_estimators=175, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')

conf = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))

plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

feature_importances = clf.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_idx], feature_importances[sorted_idx])
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()