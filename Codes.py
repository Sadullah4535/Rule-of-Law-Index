#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Try reading with a semicolon separator and ISO-8859-1 encoding
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")

# Display structure of the data
basic_info = df.info()
summary_stats = df.describe(include='all')
head_preview = df.head()

basic_info, summary_stats, head_preview


# In[3]:


import pandas as pd

# Veriyi yükle
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")

# Kolonları sadeleştir
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Sayısal kolonları al
numeric_cols = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
df_numeric = df[numeric_cols]
df


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP"
})

# Sort top 25 countries by WJP score descending (to appear at the top in the plot)
top25 = df.nlargest(25, 'WJP').sort_values('WJP', ascending=False)

# Sort bottom 25 countries by WJP score descending (as requested)
bottom25 = df.nsmallest(25, 'WJP').sort_values('WJP', ascending=False)

# Add group labels
top25['Group'] = 'Top 25'
bottom25['Group'] = 'Bottom 25'

# Combine the data
combined_df = pd.concat([top25, bottom25], axis=0)

# Plotting
plt.figure(figsize=(14, 18))
sns.set(style="whitegrid")

# Choose a more vibrant color palette (example: 'bright')
palette_choice = 'bright'

# Barplot
plot = sns.barplot(
    data=combined_df,
    x='WJP', y='Country',
    hue='Group',
    dodge=False,
    palette=palette_choice
)

# Annotate bars with WJP scores, ignoring very small values
for bar in plot.patches:
    width = bar.get_width()
    if width > 0.01:
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{width:.2f}", va='center', fontsize=9, color='black')

# Title and labels
plt.title("Top 25 and Bottom 25 Countries by Rule of Law Index Score", fontsize=14, weight='bold')
plt.xlabel("WJP Rule of Law Index Score")
plt.ylabel("Country")
plt.legend(title='Group', loc='lower right')
plt.tight_layout()
plt.show()

print(combined_df)


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data and simplify column names
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# List of variables (factors)
variables = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']

# Plot settings
sns.set(style="whitegrid")
fig, axes = plt.subplots(4, 2, figsize=(14, 16))  # 4 rows, 2 columns
axes = axes.flatten()  # Flatten to easily iterate

# Loop over each variable
for i, var in enumerate(variables):
    top10 = df.nlargest(10, var).sort_values(by=var, ascending=False)
    
    print(f"\nTop 10 Countries by {var}:\n")
    print(top10[['Country', var]].to_string(index=False))

    sns.barplot(
        ax=axes[i],
        data=top10,
        x=var,
        y='Country',
        palette='viridis'
    )

    # Annotate values
    for bar in axes[i].patches:
        width = bar.get_width()
        if width > 0.01:
            axes[i].text(width + 0.005, bar.get_y() + bar.get_height()/4,
                         f'{width:.2f}', va='center', fontsize=7, color='black')

    axes[i].set_title(f"Top 10 Countries by {var}", fontsize=9, weight='bold')
    axes[i].set_xlabel(f"{var} Score", fontsize=8)
    axes[i].set_ylabel("Country", fontsize=8)
    axes[i].tick_params(labelsize=7)

# Remove any unused axes (not needed here since 4x2=8)
plt.tight_layout()
plt.show()


# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data and simplify column names
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# List of variables (factors)
variables = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']

# Plot settings
sns.set(style="whitegrid")
fig, axes = plt.subplots(4, 2, figsize=(14, 16))  # 4 rows, 2 columns
axes = axes.flatten()  # Flatten axes array for easy iteration

# Loop over each variable for bottom 10 countries
for i, var in enumerate(variables):
    # Get bottom 10 countries by this variable
    bottom10 = df.nsmallest(10, var).sort_values(by=var, ascending=False)
    
    # Print numerical values
    print(f"\nBottom 10 Countries by {var}:\n")
    print(bottom10[['Country', var]].to_string(index=False))

    # Create horizontal bar plot
    sns.barplot(
        ax=axes[i],
        data=bottom10,
        x=var,
        y='Country',
        palette='magma'
    )
    
    # Annotate values on bars
    for bar in axes[i].patches:
        width = bar.get_width()
        if width > 0.01:
            axes[i].text(width + 0.005, bar.get_y() + bar.get_height()/4,
                         f'{width:.2f}', va='center', fontsize=7, color='black')

    # Title and labels
    axes[i].set_title(f"Bottom 10 Countries by {var}", fontsize=9, weight='bold')
    axes[i].set_xlabel(f"{var} Score", fontsize=8)
    axes[i].set_ylabel("Country", fontsize=8)
    axes[i].tick_params(labelsize=7)

# Eğer 4x2 grid'den daha az grafik varsa kalan boş subplot'ları kaldır
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and simplify column names
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Select numerical columns
numeric_cols = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
df_numeric = df[numeric_cols]

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='cool', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Rule of Law Index Factors", fontsize=14)
plt.tight_layout()
plt.show()

# Print the correlation matrix
print(corr_matrix)


# In[ ]:





# In[12]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and rename columns for simplicity
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Select numerical feature columns for PCA
features = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Add PCA results back to the dataframe
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Print explained variance ratios
explained_variance = pca.explained_variance_ratio_
print(f"PC1 explains {explained_variance[0]:.2%} of the variance")
print(f"PC2 explains {explained_variance[1]:.2%} of the variance")
print(f"Total explained variance by PC1 and PC2: {explained_variance.sum():.2%}")

# Define function to assign quadrant based on PC1 and PC2 values
def assign_quadrant(row):
    if row['PC1'] >= 0 and row['PC2'] >= 0:
        return 'Q1 (PC1+, PC2+)'
    elif row['PC1'] < 0 and row['PC2'] >= 0:
        return 'Q2 (PC1-, PC2+)'
    elif row['PC1'] < 0 and row['PC2'] < 0:
        return 'Q3 (PC1-, PC2-)'
    else:
        return 'Q4 (PC1+, PC2-)'

# Apply the quadrant assignment
df['Quadrant'] = df.apply(assign_quadrant, axis=1)

# Print summary table with Country, PC1, PC2, and Quadrant
print(df[['Country', 'PC1', 'PC2', 'Quadrant']].sort_values(by=['PC1', 'PC2'], ascending=[False, False]))

# Plot PCA scatterplot colored by WJP score
plt.figure(figsize=(14, 10))
scatter = sns.scatterplot(data=df, x='PC1', y='PC2', hue='WJP', palette='viridis', s=100, edgecolor='black')

# Annotate country names
for i in range(df.shape[0]):
    plt.text(x=df['PC1'][i] + 0.1, y=df['PC2'][i], s=df['Country'][i], fontsize=8)

plt.title(f"PCA Biplot of Rule of Law Factors\n(PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%} Variance Explained)", fontsize=16)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%} Variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%} Variance)")
plt.grid(True)
plt.legend(title="WJP Score", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:





# In[13]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and rename columns for simplicity
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",  # Eğer zaten varsa tekrar yazmaya gerek yok
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Select numerical feature columns for PCA
features = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Add PCA results back to the dataframe
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Define quadrant for each country
def assign_quadrant(row):
    if row['PC1'] >= 0 and row['PC2'] >= 0:
        return "Q1 (PC1+, PC2+)"
    elif row['PC1'] < 0 and row['PC2'] >= 0:
        return "Q2 (PC1-, PC2+)"
    elif row['PC1'] < 0 and row['PC2'] < 0:
        return "Q3 (PC1-, PC2-)"
    else:
        return "Q4 (PC1+, PC2-)"

df['Quadrant'] = df.apply(assign_quadrant, axis=1)

# Print explained variance ratios
explained_variance = pca.explained_variance_ratio_
print(f"PC1 explains {explained_variance[0]:.2%} of the variance")
print(f"PC2 explains {explained_variance[1]:.2%} of the variance")
print(f"Total explained variance by PC1 and PC2: {explained_variance.sum():.2%}\n")

print("Quadrant Definitions:")
print("Q1 (PC1+, PC2+): Countries with positive PC1 and PC2 values.")
print("Q2 (PC1-, PC2+): Countries with negative PC1 but positive PC2 values.")
print("Q3 (PC1-, PC2-): Countries with negative PC1 and PC2 values.")
print("Q4 (PC1+, PC2-): Countries with positive PC1 but negative PC2 values.\n")

# Ensure all rows are printed
pd.set_option('display.max_rows', None)

# Print sorted dataframe with countries and their PCA quadrant
print(df[['Country', 'PC1', 'PC2', 'Quadrant']].sort_values(by=['PC1', 'PC2'], ascending=[False, False]))

# Save the result to a CSV file for detailed inspection
df[['Country', 'PC1', 'PC2', 'Quadrant']].sort_values(by=['PC1', 'PC2'], ascending=[False, False]) \
  .to_csv("PCA_Countries_Quadrants.csv", index=False)

# Plot PCA scatterplot colored by WJP score
plt.figure(figsize=(14, 10))
scatter = sns.scatterplot(data=df, x='PC1', y='PC2', hue='WJP', palette='viridis', s=100, edgecolor='black')

# Annotate country names
for i in range(df.shape[0]):
    plt.text(x=df['PC1'][i] + 0.1, y=df['PC2'][i], s=df['Country'][i], fontsize=8)

plt.title(f"PCA Biplot of Rule of Law Factors\n(PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%} Variance Explained)", fontsize=16)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%} Variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%} Variance)")
plt.grid(True)
plt.legend(title="WJP Score", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Clustering

# In[ ]:





# In[15]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and preprocess data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

features = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Elbow & Silhouette
sse, sil_scores = [], []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df[['PC1', 'PC2']])
    sse.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df[['PC1', 'PC2']], labels))

# Plot elbow and silhouette (with fixed k=6)
optimal_k = 6
fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = 'tab:blue'
color2 = 'tab:green'
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia (SSE)', color=color1)
ax1.plot(k_range, sse, marker='o', color=color1, label='Inertia (SSE)')
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color=color2)
ax2.plot(k_range, sil_scores, marker='s', linestyle='--', color=color2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color2)
ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.title('Elbow and Silhouette Score for K-Means Clustering')
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
plt.tight_layout()
plt.show()

# KMeans Clustering with k=6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df[['PC1', 'PC2']])
kmeans_countries = df.groupby('KMeans_Cluster')['Country'].apply(list)

# Hierarchical Clustering with k=6
linked = linkage(df[['PC1', 'PC2']], method='ward')
df['Hierarchical_Cluster'] = fcluster(linked, t=optimal_k, criterion='maxclust') - 1
hierarchical_countries = df.groupby('Hierarchical_Cluster')['Country'].apply(list)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df[['PC1', 'PC2']])
dbscan_countries = df.groupby('DBSCAN_Cluster')['Country'].apply(list)

# Return country clusters for display
(kmeans_countries, hierarchical_countries, dbscan_countries)


# In[ ]:





# In[16]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# Prevent pandas from truncating long lists
pd.set_option('display.max_seq_item', None)

# Load and preprocess data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "Country": "Country",
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

features = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PC1'] = pca_result[:, 0]
df['PC2'] = pca_result[:, 1]

# Calculate Elbow & Silhouette scores for k=2 to 10
sse, sil_scores = [], []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df[['PC1', 'PC2']])
    sse.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df[['PC1', 'PC2']], labels))

# Set optimal k to 6 explicitly
optimal_k = 6

# Plot Elbow and Silhouette scores with fixed k=6 line
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
color2 = 'tab:green'

ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia (SSE)', color=color1)
ax1.plot(k_range, sse, marker='o', color=color1, label='Inertia (SSE)')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color=color2)
ax2.plot(k_range, sil_scores, marker='s', linestyle='--', color=color2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color2)

ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')

plt.title('Elbow and Silhouette Score for K-Means Clustering')
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
plt.tight_layout()
plt.show()

# KMeans clustering with k=6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df[['PC1', 'PC2']])

# Plot KMeans clusters with annotations
plt.figure(figsize=(12, 9))
palette = sns.color_palette("Set2", optimal_k)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='KMeans_Cluster', palette=palette, s=100, edgecolor='black')

for i in range(df.shape[0]):
    plt.text(df['PC1'][i] + 0.1, df['PC2'][i], df['Country'][i], fontsize=8)

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [f"Cluster {i} (n={sum(df['KMeans_Cluster'] == i)})" for i in range(optimal_k)]
plt.legend(handles=handles, labels=new_labels, title='KMeans Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title(f'K-Means Clustering (k={optimal_k}) on PCA Components')
plt.tight_layout()
plt.show()

kmeans_countries = df.groupby('KMeans_Cluster')['Country'].apply(list)

# Hierarchical clustering with k=6
linked = linkage(df[['PC1', 'PC2']], method='ward')
df['Hierarchical_Cluster'] = fcluster(linked, t=optimal_k, criterion='maxclust') - 1

# Scatter plot with hierarchical cluster colors (mevcut)
plt.figure(figsize=(12, 9))
palette = sns.color_palette("Set1", optimal_k)
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Hierarchical_Cluster', palette=palette, s=100, edgecolor='black')

for i in range(df.shape[0]):
    plt.text(df['PC1'][i] + 0.1, df['PC2'][i], df['Country'][i], fontsize=8)

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [f"Cluster {i} (n={sum(df['Hierarchical_Cluster'] == i)})" for i in range(optimal_k)]
plt.legend(handles=handles, labels=new_labels, title='Hierarchical Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title(f'Hierarchical Clustering (k={optimal_k}) on PCA Components')
plt.tight_layout()
plt.show()

hierarchical_countries = df.groupby('Hierarchical_Cluster')['Country'].apply(list)

# Dendrogram plot for hierarchical clustering
plt.figure(figsize=(15, 8))
dendrogram(linked, labels=df['Country'].values, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Countries')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df[['PC1', 'PC2']])

plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='DBSCAN_Cluster', palette='tab10', s=100, edgecolor='black')

for i in range(df.shape[0]):
    plt.text(df['PC1'][i] + 0.1, df['PC2'][i], df['Country'][i], fontsize=8)

dbscan_labels = df['DBSCAN_Cluster'].unique()
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [f"Cluster {label} (n={sum(df['DBSCAN_Cluster'] == label)})" for label in dbscan_labels]
plt.legend(handles=handles, labels=new_labels, title='DBSCAN Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('DBSCAN Clustering on PCA Components')
plt.tight_layout()
plt.show()

dbscan_countries = df.groupby('DBSCAN_Cluster')['Country'].apply(list)

# Print full list of countries in each cluster without truncation
print("KMeans Clusters:")
for cluster_id, countries in kmeans_countries.items():
    print(f"Cluster {cluster_id} ({len(countries)} countries):")
    for country in countries:
        print(f" - {country}")
    print()

print("Hierarchical Clusters:")
for cluster_id, countries in hierarchical_countries.items():
    print(f"Cluster {cluster_id} ({len(countries)} countries):")
    for country in countries:
        print(f" - {country}")
    print()

print("DBSCAN Clusters:")
for cluster_id, countries in dbscan_countries.items():
    print(f"Cluster {cluster_id} ({len(countries)} countries):")
    for country in countries:
        print(f" - {country}")
    print()


# In[ ]:





# In[17]:


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# K-Means metrikleri
kmeans_labels = df['KMeans_Cluster']
kmeans_silhouette = silhouette_score(df[['PC1', 'PC2']], kmeans_labels)
kmeans_ch = calinski_harabasz_score(df[['PC1', 'PC2']], kmeans_labels)
kmeans_db = davies_bouldin_score(df[['PC1', 'PC2']], kmeans_labels)

# Hiyerarşik metrikleri
hier_labels = df['Hierarchical_Cluster']
hier_silhouette = silhouette_score(df[['PC1', 'PC2']], hier_labels)
hier_ch = calinski_harabasz_score(df[['PC1', 'PC2']], hier_labels)
hier_db = davies_bouldin_score(df[['PC1', 'PC2']], hier_labels)

# DBSCAN metrikleri
dbscan_labels = df['DBSCAN_Cluster']
# DBSCAN'de -1 label'ı aykırı gözlem olduğundan bazı metrikler hata verebilir, onları çıkaralım:
core_samples_mask = dbscan_labels != -1
if sum(core_samples_mask) > 1 and len(set(dbscan_labels[core_samples_mask])) > 1:
    dbscan_silhouette = silhouette_score(df.loc[core_samples_mask, ['PC1', 'PC2']], dbscan_labels[core_samples_mask])
    dbscan_ch = calinski_harabasz_score(df.loc[core_samples_mask, ['PC1', 'PC2']], dbscan_labels[core_samples_mask])
    dbscan_db = davies_bouldin_score(df.loc[core_samples_mask, ['PC1', 'PC2']], dbscan_labels[core_samples_mask])
else:
    dbscan_silhouette = None
    dbscan_ch = None
    dbscan_db = None

# Metrikleri yazdır
print("Clustering Metrics Comparison:")
print(f"KMeans: Silhouette = {kmeans_silhouette:.3f}, Calinski-Harabasz = {kmeans_ch:.3f}, Davies-Bouldin = {kmeans_db:.3f}")
print(f"Hierarchical: Silhouette = {hier_silhouette:.3f}, Calinski-Harabasz = {hier_ch:.3f}, Davies-Bouldin = {hier_db:.3f}")
if dbscan_silhouette is not None:
    print(f"DBSCAN: Silhouette = {dbscan_silhouette:.3f}, Calinski-Harabasz = {dbscan_ch:.3f}, Davies-Bouldin = {dbscan_db:.3f}")
else:
    print("DBSCAN: Metrik hesaplanamadı (yetersiz core cluster sayısı)")


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate cluster means
cluster_means = df.groupby('KMeans_Cluster')[features].mean()

# Print cluster means numerically
print("Cluster Profile Means (KMeans):")
print(cluster_means)
print()

# Radar chart setup
labels = np.array(features)
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for cluster in cluster_means.index:
    values = cluster_means.loc[cluster].tolist()
    values += values[:1]  # complete the loop
    ax.plot(angles, values, label=f'Cluster {cluster}')
    ax.fill(angles, values, alpha=0.25)
    
    # Annotate each feature value on the plot
    for i in range(num_vars):
        angle_rad = angles[i]
        value = values[i]
        # Place the text slightly away from the data point
        ax.text(angle_rad, value + 0.05*cluster_means.max().max(), f"{value:.2f}", fontsize=8, ha='center', va='center')

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, cluster_means.max().max() * 1.2)
plt.title('Cluster Profile Radar Chart for K-Means')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()


# In[ ]:





# In[19]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal

# Load the dataset
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")

# Rename relevant columns for simplicity
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Define numerical columns to be used for clustering and analysis
numeric_cols = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)

# Perform ANOVA and Kruskal-Wallis tests for each feature across clusters
anova_results = []
kruskal_results = []

for col in numeric_cols:
    grouped_data = [group[col].values for name, group in df.groupby('KMeans_Cluster')]
    anova_stat, anova_p = f_oneway(*grouped_data)
    kruskal_stat, kruskal_p = kruskal(*grouped_data)
    anova_results.append((col, anova_p))
    kruskal_results.append((col, kruskal_p))

# Convert results to DataFrame
anova_df = pd.DataFrame(anova_results, columns=["Feature", "ANOVA_p_value"])
kruskal_df = pd.DataFrame(kruskal_results, columns=["Feature", "Kruskal_p_value"])

# Merge results into a combined table
combined_stats = pd.merge(anova_df, kruskal_df, on="Feature")
print(combined_stats)


# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


# Calculate cluster means
cluster_means = df.groupby('KMeans_Cluster')[numeric_cols].mean()

# Radar plot
import numpy as np

labels = np.array(numeric_cols)
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in cluster_means.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Radar Chart of Cluster Means")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()


# In[ ]:





# In[27]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal

# 0. Fix for MKL memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# 1. Load and preprocess data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})
numeric_cols = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']

# 2. Standardize and cluster
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])
df['KMeans_Cluster'] = KMeans(n_clusters=6, random_state=42).fit_predict(scaled_data)

# 3. Statistical Tests (ANOVA and Kruskal-Wallis)
anova_results, kruskal_results = [], []
for col in numeric_cols:
    groups = [group[col].values for _, group in df.groupby('KMeans_Cluster')]
    anova_p = f_oneway(*groups).pvalue
    kruskal_p = kruskal(*groups).pvalue
    anova_results.append((col, anova_p))
    kruskal_results.append((col, kruskal_p))

# 4. Create Table 4 - Combined statistical results
table4 = pd.merge(
    pd.DataFrame(anova_results, columns=["Feature", "ANOVA_p_value"]),
    pd.DataFrame(kruskal_results, columns=["Feature", "Kruskal_p_value"]),
    on="Feature"
)
print("Table 4: ANOVA and Kruskal-Wallis Test p-values for Each Feature Across Clusters\n")
print(table4.to_string(index=False))

# 5. Plot Figure 11 - Boxplots with means annotated
num_plots = len(numeric_cols)
cols = 3
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x='KMeans_Cluster', y=col, data=df, hue='KMeans_Cluster',
                palette='Set2', ax=ax, dodge=False, legend=False)
    
    means = df.groupby('KMeans_Cluster')[col].mean()
    for cluster_idx, mean in means.items():
        ax.text(cluster_idx, mean + df[col].max() * 0.02, f'{mean:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_title(f'{col} by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(col)

# Remove any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Figure 11: Boxplots of Legal Indicators by KMeans Cluster with Mean Values Annotated',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[28]:


from scipy.stats import shapiro

normality_results = []

for col in numeric_cols:
    for cluster_label, group in df.groupby('KMeans_Cluster'):
        stat, p_value = shapiro(group[col])
        normality_results.append({
            'Feature': col,
            'Cluster': cluster_label,
            'Shapiro_Wilk_Stat': stat,
            'p_value': p_value,
            'Normal_Distribution': p_value > 0.05  # p>0.05 ise normal dağılım varsayılabilir
        })

normality_df = pd.DataFrame(normality_results)
print(normality_df)


# In[29]:


import pandas as pd
from scipy.stats import shapiro
import io

# 1. Load and preprocess data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})
numeric_cols = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']


# Shapiro-Wilk testi
for col in numeric_cols:
    stat, p_value = shapiro(df[col])
    print(f"Feature: {col}")
    print(f"  Shapiro-Wilk test statistic: {stat:.4f}, p-value: {p_value:.4e}")
    if p_value > 0.05:
        print("  -> Data seems to be normally distributed (fail to reject H0)\n")
    else:
        print("  -> Data is not normally distributed (reject H0)\n")


# In[30]:


import pingouin as pg

# X: sadece sayısal kolonlar (örneğin df[numeric_cols])
mardia_test_result = pg.multivariate_normality(df[numeric_cols], alpha=0.05)
print(mardia_test_result)


# In[31]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.multivariate.manova import MANOVA

# Load the data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")

# Rename columns for easier access
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Define features for clustering and scaling
features = ['WJP', 'CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X_scaled = StandardScaler().fit_transform(df[features])

# Perform KMeans clustering with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Conduct MANOVA to test for multivariate differences across clusters
manova = MANOVA.from_formula('WJP + CGP + AC + OG + FR + OS + RE + CJ1 + CJ2 ~ cluster', data=df)
manova_results = manova.mv_test()

print(manova_results)


# In[42]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df = pd.read_csv("Data2024.csv", encoding="ISO-8859-1", sep=";")
df = df.rename(columns={
    "WJP Rule of Law Index: Overall Score": "WJP",
    "Factor 1: Constraints on Government Powers (CGP)": "CGP",
    "Factor 2: Absence of Corruption(AC)": "AC",
    "Factor 3: Open Government (OG)": "OG",
    "Factor 4: Fundamental Rights(FR)": "FR",
    "Factor 5: Order and Security(OS)": "OS",
    "Factor 6: Regulatory Enforcement(RE)": "RE",
    "Factor 7: Civil Justice(CJ)": "CJ1",
    "Factor 8: Criminal Justice(CJ)": "CJ2"
})

# Define features (EXCLUDING WJP!) and scale them
features = ['CGP', 'AC', 'OG', 'FR', 'OS', 'RE', 'CJ1', 'CJ2']
X_scaled = StandardScaler().fit_transform(df[features])

# KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Train Random Forest Classifier using the same features
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, df['cluster'])

# Predict cluster labels
df['Predicted_Cluster'] = rf.predict(X_scaled)

# Evaluation Metrics
accuracy = accuracy_score(df['cluster'], df['Predicted_Cluster'])
conf_mat = confusion_matrix(df['cluster'], df['Predicted_Cluster'])
r2_score = rf.score(X_scaled, df['cluster'])  # R² score

# Display prediction results
print("\n✅ Random Forest Cluster Prediction Results (first 20 rows):\n")
print(df[features + ['cluster', 'Predicted_Cluster']].head(20))

print("\n📊 Model Evaluation:")
print(f"Accuracy Score: {accuracy:.2f}")
print(f"R² Score: {r2_score:.2f}")
print("\nConfusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", classification_report(df['cluster'], df['Predicted_Cluster']))

# Feature Importance (sorted high to low)
importances = rf.feature_importances_
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Print numerical feature importance values
print("\n📌 Feature Importances (Numerical):\n")
print(feature_df.to_string(index=False))

# Plot with annotation
plt.figure(figsize=(10, 6))
bars = sns.barplot(x='Importance', y='Feature', data=feature_df, palette='husl')

for bar in bars.patches:
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va='center', fontsize=10, color='black')

plt.text(0.5, 1.10,
         f"Accuracy Score: {accuracy:.2f}    R² Score: {r2_score:.2f}",
         fontsize=12, color='darkblue', fontweight='bold', ha='center', va='center', transform=plt.gca().transAxes)

plt.title("🎯 Feature Importance for Cluster Prediction (Random Forest)", fontsize=10, weight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.xlim(0, max(feature_df['Importance']) + 0.05)
plt.tight_layout()
plt.show()


# In[ ]:




