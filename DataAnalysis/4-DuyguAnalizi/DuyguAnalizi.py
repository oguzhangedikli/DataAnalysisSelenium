import pandas as pd
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


##TEST VERİSİNİ İÇERİ ALMA.
train_df=pd.read_excel ('kullanılan_veriler/test_veri_seti.xlsx')
df=pd.read_excel('kullanılan_veriler/ogrenme_veri_seti.xlsx')
dfTAM=pd.read_excel('kullanılan_veriler/tweet.xlsx')
dataFull = pd.DataFrame(dfTAM["Tweet"])
dataFull["Tweet"] = dataFull["Tweet"].apply(lambda r: str(r))


#//////Kategorisi olmayan verileri silme
"""
df = train_df.dropna(axis=0, how = 'any')
"""


# Ana veride kategorisi olan verileri siliyor
"""
df = train_df.drop(index=['Ekonomi','Siyasi','Toplumsal'], axis=0)
"""

# Test verisinin içindekileri görme kontrol etme
"""
Testverisi = train_df.head (10)
Testverisayisi = train_df.shape
Kategoriicindekiverisayisi = train_df.groupby ("Kategori").size ()
"""








##KATEGORİLERİN SAYISALLAŞTIRILMASI

train_df['Sayısı'] = pd.factorize (train_df.Kategori)[0]
d = train_df.groupby (["Kategori", "Sayısı"]).size ()
#print(d)










##MODEL VE TEST FONKSİYONLARININ OLUŞTURULMASI

model_df = train_df[["Tweet", "Sayısı"]]
X_train, X_test, y_train, y_test = train_test_split (model_df["Tweet"], model_df["Sayısı"], test_size=0.2,
                                                     random_state=4)









##METNİN SAYISALLAŞTIRILMASI

# TF-ID Modeli

vectorizer = TfidfVectorizer ()
train_vectors = vectorizer.fit_transform (X_train)
test_vectors = vectorizer.transform (X_test)
#print(train_vectors.shape, test_vectors.shape)
#print(train_vectors)


# BOW_VECTOR Modeli
"""
BoW_Vector = CountVectorizer (min_df=0., max_df=1.)
BoW_Matrix = BoW_Vector.fit_transform (X_train)
print (BoW_Matrix)
#/// Bow_Vector CountVectorizer Bir veri setindeki kelimeleri terim sayısı matrisine dönüştürür.
"""

# Bow_Vector içindeki kelimeleri gösterme
""" 
features = BoW_Vector.get_feature_names ()
# print (features)
"""

# Sayısallaşmış test verisi içindeki kelime adedini gösterir
""" 
kelimesayisi = len (features)
print(kelimesayisi)
"""










##MODELİN EĞİTİLMESİ

#Naive Bayes

clf = MultinomialNB ()
clf.fit (train_vectors, y_train)
prediction = clf.predict (test_vectors)
#print("Naive Bayes ::\n", confusion_matrix(y_test, prediction),"\n")
#print(accuracy_score(y_test, prediction))



# Logistic Regression

LogicReg = LogisticRegression ()
LogicReg.fit (train_vectors, y_train)
prediction = LogicReg.predict (test_vectors)
#print("Logistic Regression ::\n", confusion_matrix(y_test, prediction),"\n")
#print(accuracy_score(y_test, prediction))


#DecisionTree

dTmodel = DecisionTreeClassifier ()
dTmodel.fit (train_vectors, y_train)
prediction = dTmodel.predict (test_vectors)
#print("DecisionTree ::\n", confusion_matrix(y_test, prediction),"\n")
#print(accuracy_score(y_test, prediction))


#RandomForest

rForest = RandomForestClassifier ()
rForest.fit (train_vectors, y_train)
prediction = rForest.predict (test_vectors)
# print("RandomForest ::\n",confusion_matrix(y_test,prediction),"\n")
# print(accuracy_score(y_test, prediction))


#GradientBoosting

grBoosting = GradientBoostingClassifier ()
grBoosting.fit (train_vectors, y_train)
prediction = grBoosting.predict (test_vectors)
#print("GradientBoosting ::\n", confusion_matrix(y_test, prediction), "\n")
#print(accuracy_score(y_test, prediction))


#Cross-validation
"""
scores = cross_val_score (clf, train_vectors, y_train, cv=5)
#print ("Accuracy for Naive Bayes: mean: {0:.2f} 2sd: {1:.2f}".format (scores.mean (), scores.std () * 2))
#print ("Scores::", scores)
#print ("\n")

scores2 = cross_val_score (LogicReg, train_vectors, y_train, cv=5)
#print ("Accuracy for Logistic Regression: mean: {0:.2f} 2sd: {1:.2f}".format (scores2.mean (), scores2.std () * 2))
#print ("Scores::", scores2)
print ("\n")

scores3 = cross_val_score (dTmodel, train_vectors, y_train, cv=5)
print ("Accuracy for Decision Tree: mean: {0:.2f} 2sd: {1:.2f}".format (scores3.mean (), scores3.std () * 2))
print ("Scores::", scores3)
print ("\n")

scores4 = cross_val_score (rForest, train_vectors, y_train, cv=5)
print ("Accuracy for Random Forest: mean: {0:.2f} 2sd: {1:.2f}".format (scores4.mean (), scores4.std () * 2))
print ("Scores::", scores4)
print ("\n")

scores5 = cross_val_score (grBoosting, train_vectors, y_train, cv=5)
print ("Accuracy for Gradient Boosting: mean: {0:.2f} 2sd: {1:.2f}".format (scores5.mean (), scores5.std () * 2))
print ("Scores::", scores5)
print ("\n")

methods = ["Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
accuracy = [scores.mean (), scores2.mean (), scores3.mean (), scores4.mean (), scores5.mean ()]
"""

#Uygulanan Cross-validation'ın görselleştirilmesi
"""
sns.set ()
plt.style.use('seaborn')
plt.figure (figsize=(16, 9))
plt.ylabel ("Uygulanan Algoritmalar")
plt.xlabel ("Başarı")
sns.barplot (x=accuracy, y=methods, palette="vlag")

for line in range (len (methods)):
    plt.text (0.65, line - 0.15, "{:.2f}%".format (accuracy[line] * 100), horizontalalignment='left', size='large',
              color="black")
plt.show()
"""








##MAKİNE ÖĞRENMESİ YÖNTEMİNİN KENDİ VERİMİZE UYGULANIŞI

#Çektiğimiz tweetlerin sayısallaştırılması
data = df["Tweet"]
test_vectors_ = vectorizer.transform(df["Tweet"].astype('U').values)
#print(test_vectors_.shape)
#print(test_vectors_)


#Naive Bayes İle Tahmin
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns = {0:'tahmin'}, inplace = True)
df["NaiveBayesTahmini"] = tahmin
NBtahmin= df.head(20)
#print(NBtahmin)

df.loc[df['NaiveBayesTahmini'] == 0, ['NBkategoriTahminleri']] = 'Siyasi'
df.loc[df['NaiveBayesTahmini'] == 1, ['NBkategoriTahminleri']] = 'Ekonomi'
df.loc[df['NaiveBayesTahmini'] == 2, ['NBkategoriTahminleri']] = 'Toplumsal'
NBTahminKategoriSayi=df.groupby("NBkategoriTahminleri").size()
#print(NBTahminKategoriSayi)


#Logistic Regresyon ile Tahmin
predicted = LogicReg.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns = {0:'tahmin'}, inplace = True)
df["LogisticTahmini"] = tahmin
LGtahmin= df.head(20)
#print(LGtahmin)

df.loc[df['LogisticTahmini'] == 0, ['LGkategoriTahminleri']] = 'Siyasi'
df.loc[df['LogisticTahmini'] == 1, ['LGkategoriTahminleri']] = 'Ekonomi'
df.loc[df['LogisticTahmini'] == 2, ['LGkategoriTahminleri']] = 'Toplumsal'
LGtahminKategoriSayi=df.groupby("LGkategoriTahminleri").size()
#print(LGtahminKategoriSayi)

#Kolon Sayısını Arttırmak için
pd.set_option('display.expand_frame_repr', False)

#Tahminlerin karşılaştırması
pd.set_option("max_colwidth", None)
tahminKarsilastirma=df.head(20)
#print(tahminKarsilastirma)




##TURKISH BERT ILE DUYGU ANALIZI
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer = tokenizer, model = model)

sentiment_list = []
for i in dataFull["Tweet"]:
    sentiment_list.append(sa(i))

spredict_list = []
for i in range(0, len(sentiment_list)):
    spredict_list.append(sentiment_list[i][0])

#print(sentiment_list)

spredict_list = pd.DataFrame(spredict_list)
spredict_list.head()

dataFull["label"] = spredict_list["label"]
dataFull["score"] = spredict_list["score"]
dataFull.groupby("label").size()

df["sentiment"] = dataFull["label"]
duygu_analizi = df.head(100)

print(duygu_analizi)
df.to_excel('DuyguAnalizi.xlsx')
