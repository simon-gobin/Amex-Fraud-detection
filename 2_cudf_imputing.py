#%%
import cudf
from sklearn.feature_selection import r_regression, f_regression, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import seaborn as sns
from cuml import PCA
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cupy as cp

#%%
df = cudf.read_parquet('data_reduce.parquet' )
df = df.sample(frac=0.1, random_state=0)
date = df.pop('S_2')
#%%
print('get dummies start')
df = cudf.get_dummies(df, dtype= 'Int8')
print('get dummies end')

#%%
print('imputer start')
df = df.fillna(0)
print('imputer end')

#%%
print('split data start')
y= df['target']
X = df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('split data end')

#%%
# Convert the splits back to cuDF for further GPU operations
X_train = cudf.DataFrame.from_pandas(X_train)
X_test = cudf.DataFrame.from_pandas(X_test)
y_train = cudf.Series(y_train)
y_test = cudf.Series(y_test)

# Feature Selection using correlation
correlations = r_regression(X_train.to_pandas(), y_train.to_pandas())
print('correlation done')

# ANOVA and Mutual Information
print('linear regression start')
f_values, p_values = f_regression(X_train.to_pandas(), y_train.to_pandas())
print('ANOVA start')
ANOVA_statistic, ANOVA_p_values = f_classif(X_train.to_pandas(), y_train.to_pandas())
print('start mutual info')
mutual_info = mutual_info_classif(X_train.to_pandas(), y_train.to_pandas())
print('Feature selection done')

# Store the results in a DataFrame
result_df = cudf.DataFrame({
    'Feature': X_train.columns,
    'P-value': p_values,
    'f_values': f_values,
    'Correlation': correlations,
    'ANOVA_statistic': ANOVA_statistic,
    'ANOVA_p_values': ANOVA_p_values,
    'mutual_info': mutual_info

})

#result_df.to_csv('result_df.csv')
print(result_df.head())

#%%
# Visualization (Note: For large datasets, visualize selectively to avoid memory issues)
#filtered_result_df = result_df.query('P-value <= 0.05')
#filtered_corr_df = result_df.query('Correlation >= 0.4 or `Correlation` <= -0.4')
#filtered_ANOVA_df = result_df.query('ANOVA_p_values <= 0.05')

plt.figure(figsize=(62, 10))

plt.subplot(4, 1, 1)
sns.barplot(y='f_values', x='Feature', data=result_df.to_pandas(), palette='magma')
plt.title('P-values of Features')

plt.subplot(4, 1, 2)
sns.barplot(y='Correlation', x='Feature', data=result_df.to_pandas(), palette='magma')
plt.title('Correlation of Features with Target')

plt.subplot(4, 1, 3)
sns.barplot(y='mutual_info', x='Feature', data=result_df.to_pandas(), palette='magma')
plt.title('Mutual Information Statistics of Features')

plt.subplot(4, 1, 4)
sns.barplot(y='ANOVA_statistic', x='Feature', data=result_df.to_pandas(), palette='magma')
plt.title('ANOVA Statistics of Features')

plt.tight_layout()
plt.show()
plt.savefig('mutliplot.png')

#%%
# Decision Tree Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#%%
# Predictions and Metrics
y_pred = clf.predict(X_test)

y_test_np = y_test.to_numpy()
y_pred_np = y_pred.to_numpy()

accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np)
recall = recall_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)
conf_matrix = confusion_matrix(y_test_np, y_pred_np)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

#%%
# Feature Importances
print('start feature importance')
feature_importances = clf.feature_importances_

importance_df = cudf.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

importance_df.to_csv('importance_df.csv')
print(importance_df.head())

#%%
# PCA
print('start PCA')
pca = PCA()
pca.fit(X_train)

plt.figure(figsize=(10, 6))
plt.plot(cp.asnumpy(cp.cumsum(pca.explained_variance_ratio_)), marker='o')
plt.grid(axis="both")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Principal Components")
sns.despine()
plt.show()
plt.savefig('PCA_cumulative.png')