# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Step 2: Load the Data
data = pd.read_csv('customer_segmentation_data.csv')

# Step 3: Basic Data Cleaning
# Drop duplicates and handle missing values
data = data.drop_duplicates()

# Fill missing numerical data with median values
numerical_imputer = SimpleImputer(strategy='median')
data[['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']] = numerical_imputer.fit_transform(
    data[['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']])

# Fill missing categorical data with the most frequent value (mode)
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 
                    'Preferred Communication Channel', 'Policy Type']
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

# Step 4: Feature Selection
features = data[['Age', 'Income Level', 'Coverage Amount', 'Premium Amount', 
                 'Gender', 'Marital Status', 'Education Level', 'Occupation', 
                 'Preferred Communication Channel', 'Policy Type']]

# Step 5: Preprocessing - One-hot encoding categorical features
numerical_cols = ['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)  
    ])

X = preprocessor.fit_transform(features)

# Step 6: Determine Optimal Number of Clusters using the Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 7: Fit K-means with Optimal Number of Clusters (assuming k=3)
optimal_k = 3  # Adjust based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

# Step 8: Policy Type Optimization
policy_types = data['Policy Type'].unique()

def policy_optimization():
    results = []

    for policy_type in policy_types:
        policy_data = data[data['Policy Type'] == policy_type]

        for cluster in policy_data['Cluster'].unique():
            cluster_customers = policy_data[policy_data['Cluster'] == cluster]

            if not cluster_customers.empty:
                cluster_customers_sorted = cluster_customers.sort_values(by='Premium Amount', ascending=False)

                results.append({
                    'Policy Type': policy_type,
                    'Cluster': cluster,
                    'Customer Count': len(cluster_customers_sorted),
                    'Top Customers': cluster_customers_sorted.head(10)  # Top 10 customers in each cluster
                })

    # Step 9: Sort and Display Results
    sorted_results = sorted(results, key=lambda x: x['Customer Count'], reverse=True)

    print("Summary of Clusters for Targeted Marketing by Policy Type:\n")
    for result in sorted_results:
        print(f"Policy Type: {result['Policy Type']}, Cluster {result['Cluster']} - Customer Count: {result['Customer Count']}")
        print(f"Top 10 Customers in Policy Type '{result['Policy Type']}', Cluster {result['Cluster']}:\n", result['Top Customers'])
        print("\n")

    # Additional Analysis: Identify the best policy type and cluster
    if sorted_results:
        best_result = sorted_results[0]  # Assuming the first one is the best based on customer count
        print(f"Best Policy Type and Cluster to Target: Policy Type '{best_result['Policy Type']}', Cluster {best_result['Cluster']} with {best_result['Customer Count']} customers.\n")

# Step 10: Visualization of Clusters using PCA for 2D Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Customer Segmentation Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Step 11: Repeated Input Handling
def customer_search():
    while True:
        print("\nEnter the Location:")
        Location = input()

        print("\nEnter the Policy Type:")
        Policy = input()

        # Filter by policy type and display customers
        filtered_data = data[(data['Policy Type'] == Policy)]
        
        if not filtered_data.empty:
            print(f"\nShowing Customers for Policy Type '{Policy}' in '{Location}':\n")
            print(filtered_data[['Customer ID', 'Age', 'Income Level', 'Coverage Amount', 'Premium Amount', 'Cluster']].head(10))
            
            # Ask if user wants to search again
            repeat = input("\nWould you like to search again? (yes/no): ").lower()
            if repeat != 'yes':
                break
        else:
            print(f"No customers found for Policy Type '{Policy}' in '{Location}'")

# Perform policy optimization analysis
policy_optimization()

# Start customer search with inputs
customer_search()
