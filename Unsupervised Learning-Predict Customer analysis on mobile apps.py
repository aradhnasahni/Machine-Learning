# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:47:50 2019

@author: aradhna sahni

Working Directory:
C:\Users\ashes\OneDrive\Desktop\Machine Learning
    
"""

###############################################################################
# Importing libraries and dataset
###############################################################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Importing dataset
survey_df = pd.read_excel('file:///C:/Users/ashes/OneDrive/Desktop/Final/finalExam_Mobile_App_Survey_Data_final_exam-2.xlsx')
    

 
###############################################################################
# Fundamental Dataset Exploration
###############################################################################
survey_df.columns


for col in enumerate(survey_df):
    print(col)
    
# Displaying first rows of the DataFrame
survey_df.head()

# Information about each variable
survey_df.info()

  
###############################################################################
# PCA 
###############################################################################

########################
# Step 1: Remove demographic information
########################


survey_features_reduced = survey_df.drop([    'caseID',
                                               'q1', 
                                              'q48',
                                              'q49',
                                              'q50r1', 
                                              'q50r2', 
                                              'q50r3', 
                                              'q50r4', 
                                              'q50r5', 
                                              'q54',
                                              'q55',
                                              'q56', 
                                              'q57'
                                              ],
                                                axis  = 1)   
survey_features_reduced.columns



########################
# Step 2: Scale to get equal variance
########################


scaler = StandardScaler()


scaler.fit(survey_features_reduced)

# got data on equal scales
X_scaled_step2 = scaler.transform(survey_features_reduced)



########################
# Step 3: Run PCA without limiting the number of components
########################


survey_df_pca_reduced_step3 = PCA(n_components = None,
                                  random_state = 508)


survey_df_pca_reduced_step3.fit(X_scaled_step2)


pca_factor_strengths = survey_df_pca_reduced_step3.transform(X_scaled_step2)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################


fig, ax = plt.subplots(figsize=(10, 8))

features = range(survey_df_pca_reduced_step3.n_components_)


plt.plot(features,
         survey_df_pca_reduced_step3.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()



########################
# Step 5: Run PCA again based on the desired number of components
########################

survey_df_pca_reduced_step3 = PCA(n_components = 5,
                           random_state = 508)


survey_df_pca_reduced_step3.fit(X_scaled_step2)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(survey_df_pca_reduced_step3.components_))


factor_loadings_df = factor_loadings_df.set_index(survey_features_reduced.columns)


print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings2.xlsx')



########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = survey_df_pca_reduced_step3.transform(X_scaled_step2)


X_pca_df = pd.DataFrame(X_pca_reduced) #sd how imp needs are imp for people

###############################################################################
# Combining PCA and Clustering!!!
###############################################################################

########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

from sklearn.cluster import KMeans

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['PCA 1', 'PCA2', 'PCA3','Family Oriented Needs','PCA5']


print(centroids_pca_df)


# Sending data to Excel- this file the clusters and components would be heree.
centroids_pca_df.to_excel('customers_pca_centriods3.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([survey_df.loc[ : , [
                                               'q1', 
                                              'q48',
                                              'q49',
                                              'q50r1', 
                                              'q50r2', 
                                              'q50r3', 
                                              'q50r4', 
                                              'q50r5', 
                                              'q54',
                                              'q55',
                                              'q56', 
                                              'q57']],
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))

clust_names = {0: "PCA1", 1: "PCA2", 2: "PCA3", 3: "Family Oriented Needs", 
               4: "PCA5"}

final_pca_clust_df.columns = [clust_names.get(x, x) for x in 
                              final_pca_clust_df.columns]

print(final_pca_clust_df.head(n = 5))


########################
# Step 7: Analyze in more detail 
########################

# Adding a productivity step
data_df = final_pca_clust_df



#################################
#Boxplots- Family Oriented Needs vs Behavior data
################################

# Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q1'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#Education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q48'],
            y=  data_df['Family Oriented Needs'],                
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#Marital Status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q49'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


#Children/no children
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q50r1'], #children /no children
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Children under 6/ children not under 6
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q50r2'], #children under 6 /no children
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#Children in range 6-12 and not
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q50r3'], #children under 6-12 /no children
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#Children in range 13-17 /no children
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q50r4'], #
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1


plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#children above 18 /children not above 18
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q50r5'], 
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q54'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



#Ethinicity
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q55'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# household annual income
fig, ax = plt.subplots(figsize = (40, 15))
sns.boxplot(x = survey_df['q56'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 1

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['q57'],
            y = data_df['Family Oriented Needs'],
            hue = data_df['cluster']) #cluster 4

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




#End
########################








