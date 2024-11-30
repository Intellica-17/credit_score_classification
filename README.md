# credit_score_classification
## English
This project is an example project that shows Credit Score using the data in the dataset.
### Requirements
- Python 3.x
- Jupyter Notebook or JupyterLab
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
### To install the required packages:
```Bash
pip install numpy pandas matplotlib plotly scikit-learn notebook
```
 ### Setup
 1. Clone the project repository:
 ```Bash
 git clone <repository-url>
 cd credit_score_classification
 ```
 2. Install the required libraries
 ```Bash
 pip install -r requirements.txt
 ```
 ### Usage
Run the main script to train and test the model:
 1. Run Jupyter from terminal or command line:
 ```Bash
 jupyter notebook
 ``` 
 2. In the browser window that opens, find the `.ipynb` file and click on it.
 3. Run the cells sequentially to review the analyses and graphs.

NOTE: Use the Shift + Enter key combination to run the cells in the notebook sequentially.
### Code Explanation
#### Importing Libraries
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
```
#### Loading Data Set
```Python
data = pd.read_csv('ds/credit_score_data.csv')
data.head() # Allowd us to see the first five rows
```
#### Getting information about the data
```Python
data.info()
```
#### Statistical summary information for numerical data
```Python
data.describe()
```
#### Checking for null values in the dataset
```Python
data.isnull().sum()
```
#### Cleaning unused data
```Python
data_cleaned = data.drop(columns= ["ID", "Name_surname"])
```
Create a clean dataframe by removing the unnecessary 'ID' and 'Name_surname' columns from the dataset.

#### Converting categorical data to numerical data
```Python
data_encoded = pd.get_dummies(data_cleaned, columns= ["Education_Level", "Debt_Status", "Loan_Type"], drop_first= True)
```
Converts the categorical columns 'Education_Level', 'Debt_Status', and 'Loan_Type' in the data to numerical values using the 'one-hot encoding' method. The first category (drop_first=True) is used as the reference and removed from the columns.

#### Visualization of credit score distribution
```Python
sns.countplot(x='Credit_Score', data= data_cleaned)
plt.title('Target Variable Distribution (Credit Score)')
plt.show()
```

#### Visualization of the relationship between Credit Score and Income using a box plot
```Python
px.box(data_cleaned, x= "Credit_Score", y= "Income", title= "The Relationship Between Income and Credit Score").show()
```
#### Preparing the data for evaluation and splitting the dataset into training and test sets
```Python
X = data_encoded.drop(columns=["Credit_Score"])
y = data_encoded["Credit_Score"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
#### Standardization
```Python
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)
```
We use it to scale the data.

#### Creating a Random Forest Classification Model
```Python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(xtrain_scaled, ytrain)
```

#### Prediction
```Python
ypred = rf_model.predict(xtest_scaled)
```

#### Evaluating the model's performance using a Confusion Matrix
```Python
cm = confusion_matrix(ytest, ypred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.show()
``` 
#### Visualizing the relationship between Income and predicted classes using a box plot
```Python
data_cleaned["Prediction"] = rf_model.predict(scaler.transform(data_encoded.drop(columns= ["Credit_Score"])))
fig = go.Figure()
for prediction in data_cleaned["Prediction"].unique():
    filtered_data = data_cleaned[data_cleaned["Prediction"] == prediction]
    fig.add_trace(go.Box(
        y= filtered_data["Income"], name=f"Prediction {prediction}", boxmean=True,
        marker= dict(color=np.random.choice(px.colors.qualitative.Plotly))
    ))
    fig.update_layout(
        title = "Income and Prediction Distribution",
        xaxis_title = "Prediction Class",
        yaxis_title= "Income",
        template = "plotly_white"
    )
    fig.show()
```

#### Visualizing the Credit Score Distribution
```Python
fig = go.Figure()

fig.add_trace(go.Histogram(
    x= data_cleaned['Credit_Score'],
    nbinsx=20,
    marker=dict(color='skyblue', line=dict(color='black', width=1))
))
fig.update_layout(
    title="Credit Score Distribution",
    xaxis_title="Credit Score",
    yaxis_title= "Frequency",
    bargap=0.2
)
fig.show()
```
#### Visualizing the Income Distribution and Credit Score Distribution side by side with graphs
```Python
data_cleaned[['Income', 'Credit_Score']].describe()
plt.figure(figsize=(12,6))
plt.subplot(1 , 2, 1)
plt.hist(data_cleaned['Income'], bins=30, color='skyblue', edgecolor="black")
plt.title("Income Distribution")
plt.subplot(1 , 2, 2)
plt.hist(data_cleaned['Income'], bins=20, color='salmon', edgecolor="black")
plt.title("Credit Score Distribution")
plt.show()
```

#### Visualizing the correlation between Income and Credit Score
```Python
corr_matrix = data_cleaned[['Income', 'Credit_Score']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Income and Credit Score Correlation")
plt.show()
```

#### Visualizing the relationship between Education Level and Credit Score
```Python
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Education_Level', hue='Credit_Score', palette='pastel')
plt.title('Education Level and Credit Score Relationship', fontsize=14)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.legend(title='Credit Score', labels=['Low Score (0)', 'High Score (1)'])
plt.xticks(rotation=45)
plt.show()
```
#### Visualizing the relationship between Age and Credit Score
```Python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Credit_Score', kde=True, palette='coolwarm', bins=30)
plt.title('Age and Credit Score Relationship', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.show()
```

#### Visualizing the Credit Score distribution by Education Level and Loan Type
```Python
data = pd.DataFrame({
    'Loan_Type': ['Mortgage', 'Car Loan', 'Mortgage', 'Car Loan', 'Personal Loan', 'Personal Loan', 'Mortgage'],
    'Credit_Score': [580, 720, 650, 800, 590, 780, 640],
    'Education_Level': ['High School', 'Bachelor', 'High School', 'Master', 'High School', 'Bachelor', 'Master']  
})
 
bins_score = [0, 600, 750, 850]
labels_score = ['Low Score (0-600)', 'Medium Score (600-750)', 'High Score (750-850)']
data['Credit_Score_Category'] = pd.cut(data['Credit_Score'], bins=bins_score, labels=labels_score)

cross_tab = pd.crosstab([data['Education_Level'], data['Loan_Type']], data['Credit_Score_Category'])
 
ax = cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#FF7F7F', '#FFBF7F', '#7FFF7F'])  
plt.title('Credit Score Distribution by Educational Status and Loan Type', fontsize=16)
plt.xlabel('Educational Status and Loan Type', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.xticks(rotation=45, ha='right')  
plt.legend(title='Credit Score Category', labels=labels_score, title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.show()
```

#### Printing the Accuracy Score, F1 Score, and Classification Report to the screen
```Python
accuracy = accuracy_score(ytest, ypred)
print(f"Accuracy Score: {accuracy}")

f1 = f1_score(ytest, ypred)
print(f"F1 Score {f1}")

c_report = classification_report(ytest, ypred)
print(f"Classification Report:\n {c_report}")
```

### Contributors
- Elif Nehir OĞUZ
- Cemre YAŞAR

### Licanse
This project is licensed under the [MIT Lisansı](https://opensource.org/licenses/MIT).

### NOTE
In this project, the README file is prepared in both English and Turkish. We used separator lines ('--------') between sections.

--------------------------------------------------------------------------------------------------------------------------------------------------
## Türkçe
Bu proje, verisetindeki verilerden yararlanarak Kredi Skoru sınıflandırması yapan bir örnek projedir.
### Gereksinimler
- Python 3.x
- Jupyter Notebook veya JupyterLab
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
### Gereksinimleri Yüklemek İçin
```Bash
pip install numpy pandas matplotlib plotly scikit-learn notebook
```
 ### Kurulum
 1. Proje dosyalarını indirin:
 ```Bash
 git clone <repository-url>
 cd credit_score_classification
 ```
 2. Gerekli Kütüphaneleri Yükle
 ```Bash
 pip install -r requirements.txt
 ```
 ### Kullanım
 Projenin ana dosyasını çalıştırarak modeli eğitip test edebilirsiniz:
 1. Terminalden veya komut satırından Jupyter'i çalıştırın
 ```Bash
 jupyter notebook
 ``` 
 2. Açılan tarayıcı penceresinde `.ipynb` dosyasını bulun ve üzerine tıklayın
 3. Hücreleri sırasıyla çalıştırarak analizleri ve grafikleri inceleyebilirsiniz.
NOT: Notebook'taki hücreleri sırayla çalıştırmak için Shift + Enter tuş kombinasyonu kullanabilirsiniz

### Kod Açıklaması
#### Kütüphaneleri İçe Aktarma
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
```
#### Veri setini yükleme
```Python
data = pd.read_csv('ds/credit_score_data.csv')
data.head() #İlk beş satırı görmemizi sağlar
```
#### Veriler hakkında bilgi edinme
```Python
data.info()
```
#### Sayısal verilerle ilgili istatistiksel özet
```Python
data.describe()
```
#### Veri setinde null değer olup olmadığının kontrolü
```Python
data.isnull().sum()
```
#### Kullanılmayacak verilerin temizlenmesi
```Python
data_cleaned = data.drop(columns= ["ID", "Name_surname"])
```
Veri setinden gereksiz olan "ID" ve "Name_surname" sütunlarını kaldırarak temiz bir veri çerçevesi oluşturur.

#### Kategorik verileri sayısala çevirme
```Python
data_encoded = pd.get_dummies(data_cleaned, columns= ["Education_Level", "Debt_Status", "Loan_Type"], drop_first= True)
```
Verideki Education_Level, Debt_Status ve Loan_Type kategorik sütunlarını "one-hot encoding" yöntemiyle sayısal hale getirir. İlk kategori (drop_first=True) referans olarak alınır ve sütunlardan çıkarılır.

#### Kredi skor dağlımının görselleştirilmesi
```Python
sns.countplot(x='Credit_Score', data= data_cleaned)
plt.title('Target Variable Distribution (Credit Score)')
plt.show()
```

#### Kredi Skoru ile Gelir arasındaki ilişkinin kutu grafiği ile görselleştirilmesi
```Python
px.box(data_cleaned, x= "Credit_Score", y= "Income", title= "The Relationship Between Income and Credit Score").show()
```
#### Değerlendirme için verilerin hazır hâle getirilmesi ve test verileri ile eğitim verilerinin birbirinden ayrılması
```Python
X = data_encoded.drop(columns=["Credit_Score"])
y = data_encoded["Credit_Score"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
#### Standardizasyon
```Python
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)
```
Verileri ölçeklendirmek için kullanıyoruz.

#### Rastgele Orman Sınıflandırma Modeli oluşturma
```Python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(xtrain_scaled, ytrain)
```

#### Tahminleme
```Python
ypred = rf_model.predict(xtest_scaled)
```

#### Modelin performansının Karmaşıklık Matrisi kullanarak değerlendirme
```Python
cm = confusion_matrix(ytest, ypred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.show()
``` 
#### Gelir ve tahmin edilen sınıflar arasındaki ilişkiyi kutu grafiği ile görselleştirme
```Python
data_cleaned["Prediction"] = rf_model.predict(scaler.transform(data_encoded.drop(columns= ["Credit_Score"])))
fig = go.Figure()
for prediction in data_cleaned["Prediction"].unique():
    filtered_data = data_cleaned[data_cleaned["Prediction"] == prediction]
    fig.add_trace(go.Box(
        y= filtered_data["Income"], name=f"Prediction {prediction}", boxmean=True,
        marker= dict(color=np.random.choice(px.colors.qualitative.Plotly))
    ))
    fig.update_layout(
        title = "Income and Prediction Distribution",
        xaxis_title = "Prediction Class",
        yaxis_title= "Income",
        template = "plotly_white"
    )
    fig.show()
```

#### Kredi Skor Dağılımını görselleştirme
```Python
fig = go.Figure()

fig.add_trace(go.Histogram(
    x= data_cleaned['Credit_Score'],
    nbinsx=20,
    marker=dict(color='skyblue', line=dict(color='black', width=1))
))
fig.update_layout(
    title="Credit Score Distribution",
    xaxis_title="Credit Score",
    yaxis_title= "Frequency",
    bargap=0.2
)
fig.show()
```
#### Gelir Dağılımı ve Kredi Dağılımını yan yana grafiklerle görselleştirilmesi
```Python
data_cleaned[['Income', 'Credit_Score']].describe()
plt.figure(figsize=(12,6))
plt.subplot(1 , 2, 1)
plt.hist(data_cleaned['Income'], bins=30, color='skyblue', edgecolor="black")
plt.title("Income Distribution")
plt.subplot(1 , 2, 2)
plt.hist(data_cleaned['Income'], bins=20, color='salmon', edgecolor="black")
plt.title("Credit Score Distribution")
plt.show()
```

#### Gelir ve Kredi Skoru Arasındaki Korelasyonun Görselleştirilmesi
```Python
corr_matrix = data_cleaned[['Income', 'Credit_Score']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Income and Credit Score Correlation")
plt.show()
```

#### Eğitim Düzeği ve Kredi Skoru arasındaki ilişkiyi görselleştirme
```Python
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Education_Level', hue='Credit_Score', palette='pastel')
plt.title('Education Level and Credit Score Relationship', fontsize=14)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.legend(title='Credit Score', labels=['Low Score (0)', 'High Score (1)'])
plt.xticks(rotation=45)
plt.show()
```
#### Yaş ve Kredi Skoru arasındaki ilişkinin görselleştirilmesi
```Python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Credit_Score', kde=True, palette='coolwarm', bins=30)
plt.title('Age and Credit Score Relationship', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.show()
```

#### Öğrenim Durumu ve Kredi Türüne Göre Kredi Puanı Dağılımının görselleştirilmesi
```Python
data = pd.DataFrame({
    'Loan_Type': ['Mortgage', 'Car Loan', 'Mortgage', 'Car Loan', 'Personal Loan', 'Personal Loan', 'Mortgage'],
    'Credit_Score': [580, 720, 650, 800, 590, 780, 640],
    'Education_Level': ['High School', 'Bachelor', 'High School', 'Master', 'High School', 'Bachelor', 'Master']  
})
 
bins_score = [0, 600, 750, 850]
labels_score = ['Low Score (0-600)', 'Medium Score (600-750)', 'High Score (750-850)']
data['Credit_Score_Category'] = pd.cut(data['Credit_Score'], bins=bins_score, labels=labels_score)

cross_tab = pd.crosstab([data['Education_Level'], data['Loan_Type']], data['Credit_Score_Category'])
 
ax = cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#FF7F7F', '#FFBF7F', '#7FFF7F'])  
plt.title('Credit Score Distribution by Educational Status and Loan Type', fontsize=16)
plt.xlabel('Educational Status and Loan Type', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.xticks(rotation=45, ha='right')  
plt.legend(title='Credit Score Category', labels=labels_score, title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.show()
```

#### Doğruluk Skorunun, F1 Skorunun ve Sınıflandırma Raporunun ekrana yazdırılması
```Python
accuracy = accuracy_score(ytest, ypred)
print(f"Accuracy Score: {accuracy}")

f1 = f1_score(ytest, ypred)
print(f"F1 Score {f1}")

c_report = classification_report(ytest, ypred)
print(f"Classification Report:\n {c_report}")
```
### Katkıda Bulunanlar
- Elif Nehir OĞUZ
- Cemre YAŞAR

### Lisans
Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.

### NOT
Bu projede, README dosyasını hem İngilizce hem de Türkçe olarak iki dilde hazırladık. Bölümler arasında ayırıcı çizgiler ('------') kullandık.
