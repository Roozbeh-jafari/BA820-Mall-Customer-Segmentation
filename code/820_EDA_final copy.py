#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[4]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
plt.style.use('ggplot')


# ## Load dataset

# In[5]:


df=pd.read_csv('Mall_Customers.csv')


# ## Basic Exploration to get better understanding of the dataset

# In[6]:


df.head()


# data check

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.shape


# > droping uninformative column(CustomerID) and set it as index 

# > No null value based on info
# 

# > check to see if any duplicate value exist

# In[10]:


df.set_index("CustomerID", inplace=True)


# In[11]:


df.columns = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore']


# In[12]:


df.head()


# In[13]:


df.duplicated().sum()


# In[14]:


df.describe().T


# In[15]:


df.sort_values(by="AnnualIncome")[-10:]


# In[16]:


x=df.groupby(['Gender'])['Gender'].count()
y=len(df)
r=((x/y)).round(2)

mf_ratio = pd.DataFrame(r).T
mf_ratio


# it seems there is some outliers at AnnualIncome from #195

# ## Exploratory Data Analysis

# In[17]:


fig = plt.figure()
sns.heatmap(df.corr(), annot=df.corr())
plt.yticks(rotation=45)
plt.show()


# there is not a highly correlated features, the most is between age and spendingscore(negative)

# In[18]:


plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
sns.distplot(df.Age[df['Gender']=='Female'], color='orange', hist=True, kde=True, label='Female')
sns.distplot(df.Age[df['Gender']=='Male'], color='blue', hist=True, kde=True, label='Male')
plt.title('Age')

plt.subplot(1,3,2)
sns.distplot(df.AnnualIncome[df['Gender']=='Female'], color='orange', hist=False, kde=True, label='Female')
sns.distplot(df.AnnualIncome[df['Gender']=='Male'], color='blue', hist=False, kde=True, label='Male')
plt.title('AnnualIncome')

plt.subplot(1,3,3)
sns.distplot(df.SpendingScore[df['Gender']=='Female'], color='orange', hist=False, kde=True, label='Female')
sns.distplot(df.SpendingScore[df['Gender']=='Male'], color='blue', hist=False, kde=True, label='Male')
plt.title('SpendingScore;')

plt.show()


# In[19]:


plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
sns.boxplot(x=df.Gender, y=df.Age)
plt.title('Age')

plt.subplot(1,3,2)
sns.boxplot(x=df.Gender, y=df.AnnualIncome)
plt.title('AnnualIncome')

plt.subplot(1,3,3)
sns.boxplot(x=df.Gender, y=df.SpendingScore)
plt.title('SpendingScore')

plt.show()


# The left plot indicates wider age range in male and also higher average age.
# by exploring at center and right plots, show even though men have slightly higher income but in terms of spending women spent slighty more than men

# In[20]:


plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'AnnualIncome' , 'SpendingScore']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 15)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[21]:


sns.pairplot(df, corner=True, vars = ['Age', 'AnnualIncome', 'SpendingScore'], hue = "Gender")


# In[22]:


plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'AnnualIncome' , 'SpendingScore']:
    for y in ['Age' , 'AnnualIncome' , 'SpendingScore']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


# In[23]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.scatterplot(x=df.Age, y=df.AnnualIncome, hue=df.Gender)
plt.title('Age vs AnnualIncome')

plt.subplot(1,3,2)
sns.scatterplot(x=df.Age, y=df.SpendingScore, hue=df.Gender)
plt.title('Age vs SpendingScore')

plt.subplot(1,3,3)
sns.scatterplot(x=df.AnnualIncome, y=df.SpendingScore, hue=df.Gender)
plt.title('AnnualIncome vs SpendingScore')

plt.show()


# Have you found something?
# I did actually... you can see there seems to be 2 groups of customers by age vs score (top left quarter vs bottom right quarter), where diagonal is delimiting them.
# 
# What is more important is actually chart Income vs Score where we can see 5 different groups of customers (corners & center). What does it mean? We've probably found ideal way to cluster our customers based on income and score!
# 
# Also based on Annualincom vs SpendingScore, it is kind of a rabbit head, and indicates 5 groups
# 

# In[24]:


df.head()


# In[25]:


df1 = df.replace("Male", 0)
df1 = df1.replace("Female", 1)


# ### Standardization 

# In[26]:


# scale, because clearly these are not on the same scale, and I want to ensure each variable has equal weight
sc_org = StandardScaler()
xs_org = sc_org.fit_transform(df1)
df_std = pd.DataFrame(xs_org, index=df1.index, columns=df1.columns)


# ## Hierarchical Clustering

# In[27]:


dff2 = df.drop('Gender', axis=1)

# scale, because clearly these are not on the same scale, and I want to ensure each variable has equal weight
sc2 = StandardScaler()
xs2 = sc2.fit_transform(dff2)
dff2 = pd.DataFrame(xs2, index=dff2.index, columns=dff2.columns)


# In[28]:


# hierarchical clustering--'single', 'complete', 'average', and 'ward' methods

METHODS = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(15,5))

for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(dff2, method=m),
             labels = dff2.index,
             leaf_rotation=90,
             leaf_font_size=10)
plt.show()


# In[29]:


df_copy = df.copy()

# try to slice using the complete method
df_hccomp = linkage(df_std, method = 'complete')

# slice up clusters
df_hccomp4 = fcluster(df_hccomp, 4, criterion = 'distance')
df_copy['hccomp4'] = df_hccomp4
df_copy.hccomp4.value_counts()


# In[30]:


# try to slice using the average method
df_hcavg = linkage(df_std, method = 'average')

# slice up clusters
df_hcavg2 = fcluster(df_hcavg, 2, criterion = 'distance')
df_copy['hcavg2'] = df_hcavg2
df_copy.hcavg2.value_counts()


# In[31]:


# try to slice using the ward method
df_hcward = linkage(dff2, method = 'ward')

# slice up clusters
df_hcward15 = fcluster(df_hcward, 15, criterion = 'distance')
df_copy['hcward15'] = df_hcward15
df_copy.hcward15.value_counts()


# In[32]:


plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
sns.scatterplot(x=df_copy.Age, y=df_copy.AnnualIncome, hue=df_copy.hccomp4)
plt.title('Age vs AnnualIncome - complete')

plt.subplot(3,3,2)
sns.scatterplot(x=df_copy.Age, y=df_copy.SpendingScore, hue=df_copy.hccomp4)
plt.title('Age vs SpendingScore - complete')

plt.subplot(3,3,3)
sns.scatterplot(x=df_copy.AnnualIncome, y=df_copy.SpendingScore, hue=df_copy.hccomp4)
plt.title('AnnualIncome vs SpendingScore - complete')

plt.subplot(3,3,4)
sns.scatterplot(x=df_copy.Age, y=df_copy.AnnualIncome, hue=df_copy.hcavg2)
plt.title('Age vs AnnualIncome - average')

plt.subplot(3,3,5)
sns.scatterplot(x=df_copy.Age, y=df_copy.SpendingScore, hue=df_copy.hcavg2)
plt.title('Age vs SpendingScore - average')

plt.subplot(3,3,6)
sns.scatterplot(x=df_copy.AnnualIncome, y=df_copy.SpendingScore, hue=df_copy.hcavg2)
plt.title('AnnualIncome vs SpendingScore - average')

plt.subplot(3,3,7)
sns.scatterplot(x=df_copy.Age, y=df_copy.AnnualIncome, hue=df_copy.hcward15)
plt.title('Age vs AnnualIncome - ward')

plt.subplot(3,3,8)
sns.scatterplot(x=df_copy.Age, y=df_copy.SpendingScore, hue=df_copy.hcward15)
plt.title('Age vs SpendingScore - ward')

plt.subplot(3,3,9)
sns.scatterplot(x=df_copy.AnnualIncome, y=df_copy.SpendingScore, hue=df_copy.hcward15)
plt.title('AnnualIncome vs SpendingScore - ward')

plt.show()


# In[33]:


df_copy.groupby('hccomp4').agg('mean').iloc[:, :3]


# In[34]:


df_copy.groupby('hcavg2').agg('mean').iloc[:, :3]


# In[35]:


df_copy.groupby('hcward15').agg('mean').iloc[:, :3]


# ## K-Means Clustering

# ### 1. clusters with original dataset

# In[36]:


df_std.head()


# In[37]:


# Kmeans
KS = range(2, 10)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(df_std)
  labs = km.predict(df_std)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(df_std, labs))


# In[38]:


plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)
plt.axvline(4, color="red", linestyle= '--')

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)


plt.show();


# ### 2.Segmentation using Age and Spending Score

# In[39]:


df2 = df_std[["Age", "SpendingScore"]]
df2.head()


# In[40]:


# Kmeans
KS = range(2, 10)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(df2)
  labs = km.predict(df2)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(df2, labs))


# In[41]:


plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)
# plt.axvline(4, color="red", linestyle= '--')

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)


plt.show();


# In[42]:


X1=df.iloc[:,[1,2,3]].values


# In[43]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X1)


# In[44]:


plt.scatter(X1[y_kmeans==0,0],X1[y_kmeans==0,1],s=100,c='magenta',label='Low spenders ')
plt.scatter(X1[y_kmeans==1,0],X1[y_kmeans==1,1],s=100,c='blue',label='Young High Spenders')
plt.scatter(X1[y_kmeans==2,0],X1[y_kmeans==2,1],s=100,c='green',label='Young Average Spenders')
plt.scatter(X1[y_kmeans==3,0],X1[y_kmeans==3,1],s=100,c='cyan',label='Old Average Spenders')
#plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='burlywood',label='Sensible')
#plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='blue',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.ioff()
plt.show
pass


# So we can cluster the data into four groups.
# 
# 1.Low spenders
# 
# 2.Young High Spenders
# 
# 3.Young Average Spenders
# 
# 4.Old Average spenders
# 
# We can clearly see that Only young people(18-40 age group) are involved in High Spending.As age increases people fall into average or Low spending catogery.

# In[45]:


for i, s in enumerate(silo[:10]):
  print(i+2,s) # +2 to align num clusters with value


# In[46]:


# get the model
k6 = KMeans(6)
k6_labs = k6.fit_predict(df2)

# metrics
k6_silo = silhouette_score(df2, k6_labs)
k6_ssamps = silhouette_samples(df2, k6_labs)
np.unique(k6_labs)


# ### 3.Segmentation using Annual Income and Spending Score

# In[47]:


df3 = df_std[["AnnualIncome", "SpendingScore"]]
df3.head()


# In[48]:


# scale, because clearly these are not on the same scale, and I want to ensure each variable has equal weight
sc = StandardScaler()
xs = sc.fit_transform(df3)
df3 = pd.DataFrame(xs, index=df3.index, columns=df3.columns)


# In[49]:


# Kmeans
KS = range(2, 10)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(df3)
  labs = km.predict(df3)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(df3, labs))


# In[50]:


plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)
plt.axvline(5, color="red", linestyle= '--')

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)


plt.show();


# In[51]:


df.head()


# In[53]:


X=df.iloc[:,[2,3]].values


# In[54]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[55]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


# In[56]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='magenta',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='yellow',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='burlywood',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show
pass


# ### Cluster 1- High income low spending =Careful
# 
# ### Cluster 2- Medium income medium spending =Standard
# 
# ### Cluster 3- High Income and high spending =Target
# 
# ### Cluster 4- Low Income and high spending =Careless
# 
# ### Cluster 5- Low Income and low spending =Sensible

# ### 4.Segmentation using Age and Income

# In[57]:


X2=df.iloc[:,[1,2]].values


# In[58]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[59]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X2)


# In[60]:


plt.scatter(X2[y_kmeans==0,0],X2[y_kmeans==0,1],s=100,c='magenta',label='High Earners ')
plt.scatter(X2[y_kmeans==1,0],X2[y_kmeans==1,1],s=100,c='blue',label='Young Low Earners')
plt.scatter(X2[y_kmeans==2,0],X2[y_kmeans==2,1],s=100,c='green',label='Average Earners')
plt.scatter(X2[y_kmeans==3,0],X2[y_kmeans==3,1],s=100,c='cyan',label='Old Average Earners')
plt.scatter(X2[y_kmeans==4,0],X2[y_kmeans==4,1],s=100,c='burlywood',label='Old Low Earners ')
plt.scatter(X2[y_kmeans==5,0],X2[y_kmeans==5,1],s=100,c='yellow',label='Young Average Earners')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Age')
plt.ylabel('Annual income')
plt.legend()
plt.ioff()
plt.show
pass


# We can see people in age group 0f 30-40 have high number of high income people

# ### PCA

# In[61]:


# scale, because clearly these are not on the same scale, and I want to ensure each variable has equal weight
sc = StandardScaler()
xs = sc.fit_transform(df1)
df_std = pd.DataFrame(xs, index=df1.index, columns=df1.columns)


# In[62]:


pca = PCA(.9)
pca.fit(df_std)
pcs = pca.transform(df_std) #output is array, we can transforme it to DataFrame
pcs[:3]


# In[63]:


df_pca = pd.DataFrame(pcs)
df_pca.head() #all 4 components


# In[64]:


num_comp = range(pca.n_components_)
variance = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_



plt.bar(num_comp, variance)
plt.xticks(num_comp)
plt.ylabel("Variance")
plt.xlabel("Principal components")
plt.show()


# In[65]:


sns.lineplot(num_comp, variance_ratio)
plt.show();


#  we have decided to keep all our components

# In[66]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(df_pca)


# In[67]:


df["labs"] = y_kmeans
df.head()


# In[68]:


df.groupby("labs").describe()


# In[69]:


dff = df.copy()


# In[70]:


dff.labs.replace(4, "Highest", inplace=True)
dff.labs.replace(0, "High", inplace=True)
dff.labs.replace(2, "Average", inplace=True)
dff.labs.replace(3, "Low", inplace=True)
dff.labs.replace(1, "Lowest", inplace=True)


# In[71]:


dff.groupby("labs").describe()


# In[72]:


clusters = ["Hieghest", "High", "Average", "Low", "lowest"]
distribution = [19.5, 28, 14, 24, 14.5]
dist = pd.DataFrame({"Clusters":clusters, "Distribution%": distribution})


# In[73]:


dist


# ## tsne

# In[74]:


tsne = TSNE() #always in real word n_components are 2
tsne.fit(pcs)


# In[75]:


# get the embeddings
te = tsne.embedding_

#
# the shape
te.shape


# In[76]:


te_df = pd.DataFrame(te, columns=["A", "B"])
te_df.head()


# In[77]:


# PAL = sns.color_palette("bright", 10) 
plt.figure(figsize=(10, 8))
sns.scatterplot(x= "A" , y="B", hue=y_kmeans, data=te_df, legend="full")


# In[ ]:




