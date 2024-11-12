#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()


# In[2]:


soildata=pd.read_csv(r"D:\SRM\Big data\Data Orch\archive (3)\data.csv")


# In[3]:


soildata.rename(columns = {'moisture':'Humidité'}, inplace = True)
soildata.rename(columns = {'temp':'Température'}, inplace = True)
soildata.rename(columns = {'pump':'Arrosage'}, inplace = True)
print( 'Taille(n_lignes,n_colonnes)of data frame :',soildata.shape)
soildata.head(10)


# In[4]:


soildata.describe()


# In[5]:


soildata['Arrosage'].value_counts()


# In[6]:


labels=["arrosed","non arrosed"]
y=np.array([150, 50])
plt.pie(y,labels=labels,autopct='%1.1f%%')
plt.title('arrosage distrubution')


# In[7]:


plt.scatter(soildata.Humidité,soildata.Arrosage,marker='+',color='blue')


# In[8]:


plt.scatter(soildata.Température,soildata.Arrosage,marker='+',color='blue')


# In[9]:


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(soildata.Humidité, soildata.Température, soildata.Arrosage,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=soildata.Arrosage)
plt.xlabel("Humidité")
plt.ylabel("Température")

plt.show()


# In[10]:


cormat = soildata.select_dtypes(include='number').corr()
cormat = round(cormat, 2)
sns.heatmap(cormat)


# In[11]:


from sklearn.model_selection import train_test_split 


# In[12]:


X = soildata.drop(['Arrosage','crop'], axis=1)
y = soildata['Arrosage']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2,stratify=y, random_state=42)


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred), ": is the accuracy score")
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred), ": is the precision score")
from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred), ": is the recall score")
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred), ": is the f1 score")


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, y_pred, labels=[1,0]))


# In[17]:


cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['arrosed','non arrosed'],normalize= False,  title='Confusion matrix')


# In[18]:


print (classification_report(y_test, y_pred))


# In[19]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import utils


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import utils


# In[22]:


datainput = pd.read_csv(r"D:\SRM\Big data\datafile.csv", delimiter = ',')


# In[23]:


#preprocessing
Profit = (datainput.iloc[:,5]*datainput.iloc[:,6]-(datainput.iloc[:,2]+datainput.iloc[:,3]+(datainput.iloc[:,5]*datainput.iloc[:,4]))).values
Profit = Profit.reshape(49,1)
Profitcopy = (datainput.iloc[:,5]*datainput.iloc[:,6]-(datainput.iloc[:,2]+datainput.iloc[:,3]+(datainput.iloc[:,5]*datainput.iloc[:,4]))).values


# In[24]:


for i in range (0,49):
    if Profit[i][0]>0:
        Profit[i][0] = 1
    else:
        Profit[i][0] = 0
X = datainput[['Crop', 'State', 'Cost of Cultivation (`/Hectare) A2+FL', 
               'Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2', 'Support price']].values

       


# In[25]:


#label encoder to categorical data 
labelencoder_X = preprocessing.LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,1] = labelencoder_X.fit_transform(X[:, 1])


# In[26]:


#One hot encoder 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder='passthrough')   
x2 = np.array(columnTransformer.fit_transform(X), dtype = float) 

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[10])], remainder='passthrough')  
x3 = np.array(columnTransformer.fit_transform(x2), dtype = float) 


# In[27]:


#output col in y 
y = Profit

#Splitting
X_train, X_test, y_train, y_test = train_test_split(x3, y, test_size=0.3, random_state=3)



# In[ ]:


#Clustering 
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x3)
    wcss.append(kmeans.inertia_)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[ ]:


plt.plot(range(1, 30), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x3)


# In[ ]:


#Plot 1. Crops in percentage
Crops = datainput.iloc[:,0]
CropsCount = {} 
for crop in Crops: 
    CropsCount[crop] = CropsCount.get(crop, 0)+1
    
#extract values and keys of dict:CropsCount
labels = list(CropsCount.keys())
values = list(CropsCount.values())

plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[ ]:


#Plot 2. State distribution
States = datainput.iloc[:,1]
StatesCount = {} 
for state in States: 
    StatesCount[state] = StatesCount.get(state, 0)+1
    
#extract values and keys of dict:CropsCount
labels = list(StatesCount.keys())
values = list(StatesCount.values())

plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()  


# In[ ]:


#Plot 3. Yield per Crop
plt.tick_params(labelsize=8)
datainput.groupby("Crop")["Yield (Quintal/ Hectare) "].agg("sum").plot.bar()


# In[ ]:




