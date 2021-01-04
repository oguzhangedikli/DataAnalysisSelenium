import pandas as pd
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from PIL import Image
import matplotlib.pyplot as plt
sw = stopwords.words("turkish")
import seaborn

"""
#VERİLERİ TEMİZLEME
veri = pd.read_excel ("KİRLİVERİ.xlsx" )
veri['Tweet']= veri['Tweet'].str.replace (r'http\S+', '')
veri['Tweet']= veri['Tweet'].str.replace (r'@\S+', '')
veri['Tweet']= veri['Tweet'].str.replace (r'#\S+', '')
veri['Tweet']= veri['Tweet'].str.replace('[^\w\s]','')
veri['Tweet']= veri['Tweet'].str.replace('\d','')         
veri['Tweet']= veri['Tweet'].str.replace('gt', '')        

#HARFLERİ KÜÇÜLTME
veri['Tweet'] = veri['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#STOPWORDS İŞLEMİ
veri['Tweet'] = veri['Tweet'].apply (lambda x: " ".join (x for x in x.split () if x not in sw))

#Sık geçen kelimelerin listelenmesi
freq_df = veri["Tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
freq_df.columns = ["kelimeler", "frekanslar"]
print(freq_df.head())
print("En sık geçen kelimelerin görselleştirilmesi : \n")
a = freq_df[freq_df.frekanslar > freq_df.frekanslar.mean() + freq_df.frekanslar.std()]
plt.bar(a["kelimeler"].head(5), a["frekanslar"].head(5))
plt.show()

#KELİME BULUTU
Tweet = " ".join(i for i in veri.Tweet)
print("Kelime bulutunun oluşturulması : \n")
wordcloud = WordCloud(background_color = "black").generate(Tweet)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

#TEMİZ VERİ DIŞA AKTARMA
veri.to_excel ("TEMİZVERİ.xlsx")
"""






train_df = pd.read_excel("3-ÖĞRENME/KATEGORİLİSET.xlsx")
train_df.head(10)
train_df.shape
train_df.groupby("Kategori").size()

#SAYISALLAŞTIRMA
train_df['labels'] = pd.factorize(train_df.Kategori)[0]
train_df.groupby(["Kategori", "labels"]).size()
print(train_df)

#Model ve Test Fonksiyonlarının Oluşturulması
from sklearn.model_selection import train_test_split
model_df = train_df[["Tweet", "labels"]]
X_train, X_test, y_train, y_test = train_test_split(model_df["Tweet"], model_df["labels"], test_size = 0.2, random_state = 4)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

#Metnin Sayısallaştırılması
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)
print(train_vectors)

#Modelin Eğitilmesi
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns

clf = MultinomialNB()
clf.fit(train_vectors, y_train)
prediction = clf.predict(test_vectors)
print("Naive Bayes ::\n", confusion_matrix(y_test, prediction),"\n")
print(accuracy_score(y_test, prediction))

LogicReg = LogisticRegression()
LogicReg.fit(train_vectors, y_train)
prediction = LogicReg.predict(test_vectors)
print("Logistic Regression ::\n", confusion_matrix(y_test, prediction),"\n")
print(accuracy_score(y_test, prediction))

dTmodel = DecisionTreeClassifier()
dTmodel.fit(train_vectors, y_train)
prediction = dTmodel.predict(test_vectors)
print("DecisionTree ::\n", confusion_matrix(y_test, prediction),"\n")
print(accuracy_score(y_test, prediction))

rForest = RandomForestClassifier()
rForest.fit(train_vectors,y_train)
prediction=rForest.predict(test_vectors)
print("RandomForest ::\n",confusion_matrix(y_test,prediction),"\n")
print(accuracy_score(y_test, prediction))

grBoosting = GradientBoostingClassifier()
grBoosting.fit(train_vectors, y_train)
prediction = grBoosting.predict(test_vectors)
print("GradientBoosting ::\n", confusion_matrix(y_test, prediction), "\n")
print(accuracy_score(y_test, prediction))

xgboost = XGBClassifier()
xgboost.fit(train_vectors,y_train)
prediction=xgboost.predict(test_vectors)
print("xgboost ::\n",confusion_matrix(y_test,prediction), "\n")
print(accuracy_score(y_test, prediction))

#CROSSVALİDATİON
scores = cross_val_score(clf, train_vectors, y_train, cv = 5)
print("Accuracy for Naive Bayes: mean: {0:.2f} 2sd: {1:.2f}".format(scores.mean(),scores.std() * 2))
print("Scores::",scores)
print("\n")

scores2 = cross_val_score(LogicReg, train_vectors, y_train, cv = 5)
print("Accuracy for Logistic Regression: mean: {0:.2f} 2sd: {1:.2f}".format(scores2.mean(),scores2.std() * 2))
print("Scores::",scores2)
print("\n")

scores3 = cross_val_score(dTmodel,train_vectors,y_train,cv=5)
print("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format(scores3.mean(),scores3.std() * 2))
print("Scores::",scores3)
print("\n")

scores4 = cross_val_score(rForest,train_vectors,y_train,cv=5)
print("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format(scores4.mean(),scores4.std() * 2))
print("Scores::",scores4)
print("\n")

scores5 = cross_val_score(grBoosting,train_vectors,y_train,cv=5)
print("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format(scores5.mean(),scores5.std() * 2))
print("Scores::",scores5)
print("\n")

scores6 = cross_val_score(xgboost, train_vectors, y_train,cv = 5)
print("Accuracy for Xgboost: mean: {0:.2f} 2sd: {1:.2f}".format(scores6.mean(),scores6.std() * 2))
print("Scores::",scores6)
print("\n")

methods = ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
accuracy = [scores.mean(), scores2.mean(), scores3.mean(), scores4.mean(), scores5.mean(), scores6.mean()]

sns.set()
plt.figure(figsize = (16, 9))
plt.ylabel("Uygulanan Algoritmalar")
plt.xlabel("Başarı")
sns.barplot(x = accuracy, y = methods, palette = "vlag")


for line in range(len(methods)):
     plt.text(0.65, line-0.15, "{:.2f}%".format(accuracy[line]*100), horizontalalignment = 'left', size = 'large', color = "black")

#plt.show()


#TEMİZ VERİLERİN SAYISALLAŞTIRILMASI
df = pd.read_excel("3-ÖĞRENME/KATEGORİSİZSET.xlsx")
df['Tweet']
test_vectors_ = vectorizer.transform(df["Tweet"].astype('U').values)
print(test_vectors_.shape)
print(test_vectors_)

#LOGİSTİC TAHMİN
predicted = LogicReg.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns = {0:'tahmin'}, inplace = True)
df["tahmin_logistic"] = tahmin
df.head(20)

df.loc[df['tahmin_logistic'] == 0, ['tahmin_category_logistic']] = 'Çevre'
df.loc[df['tahmin_logistic'] == 1, ['tahmin_category_logistic']] = 'Ekonomi'
df.loc[df['tahmin_logistic'] == 2, ['tahmin_category_logistic']] = 'Siyaset'

df.head(20)
df.groupby("tahmin_category_logistic").size()


#NAİVE BAYES TAHMİN
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns = {0:'tahmin'}, inplace = True)
df["tahmin_naive_bayes"] = tahmin

df.head(20)

df.loc[df['tahmin_naive_bayes'] == 0, ['tahmin_category_nb']] = 'Çevre'
df.loc[df['tahmin_naive_bayes'] == 1, ['tahmin_category_nb']] = 'Ekonomi'
df.loc[df['tahmin_naive_bayes'] == 2, ['tahmin_category_nb']] = 'Siyaset'

df.groupby("tahmin_category_nb").size()

#Algoritma Tahmin Sonuçlarının Karşılaştırılması
pd.set_option("max_colwidth", None)
df.head(20)

df.groupby("tahmin_category_nb").size()

#df.to_excel('MAKİNEOGRENMESONUC.xlsx')


#DUYGU ANALİZİ

data = pd.DataFrame(df["Tweet"])

data.head()
data["Tweet"] = data["Tweet"].apply(lambda r: str(r))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer = tokenizer, model = model)

sentiment_list = []
for i in data["Tweet"]:
    sentiment_list.append(sa(i))

sentiment_list

print(sentiment_list[4][0])
spredict_list = []
for i in range(0, len(sentiment_list)):
    spredict_list.append(sentiment_list[i][0])

spredict_list

spredict_list = pd.DataFrame(spredict_list)

spredict_list.head()
data["label"] = spredict_list["label"]
data["score"] = spredict_list["score"]
data.head(10)
data.groupby("label").size()
df.head()
data.head()
df["sentiment"] = data["label"]
df.head()

df.drop_duplicates(subset=["Tweet"], keep=False,inplace=False)

#GORSELLEŞTIRME


#LOGISTIC
grup_logis = df.groupby(["tahmin_category_logistic", "sentiment"]).size()
grup_logis = pd.DataFrame(grup_logis).reset_index()

grup_logis
grup_logis.rename(columns = {0:'tweet_sayisi'}, inplace = True)
sns.catplot(x = "tahmin_category_logistic", y = "tweet_sayisi", hue = "sentiment", kind = "bar", data = grup_logis)
plt.show()

#NAİVEBAYES

grup_nb = df.groupby(["tahmin_category_nb", "sentiment"]).size()
grup_nb = pd.DataFrame(grup_nb).reset_index()
grup_nb
grup_nb.rename(columns = {0:'tweet_sayisi'}, inplace = True)
sns.catplot(x = "tahmin_category_nb", y = "tweet_sayisi", hue = "sentiment", kind = "bar", data = grup_nb)
plt.show()
