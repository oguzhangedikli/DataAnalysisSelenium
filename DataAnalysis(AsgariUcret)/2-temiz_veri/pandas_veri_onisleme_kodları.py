import pandas as pd
import numpy as np
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


data = pd.read_excel ("kullanılan_veriler/kirli_veri.xlsx" )

"""
data['Tweet']= data['Tweet'].str.replace (r'http\S+', '') #////Tweetlerin içindeki linkleri temizler.
data['Tweet']= data['Tweet'].str.replace (r'@\S+', '')    #////Tweetlerin içindeki Etiketlemeleri temizler.
data['Tweet']= data['Tweet'].str.replace (r'#\S+', '')    #////Tweetlerin içindeki Hastagleri temizler.
data['Tweet']= data['Tweet'].str.replace('[^\w\s]','')    #////Tweetlerin içindeki Noktalama İşaretlerini temizler.
data['Tweet']= data['Tweet'].str.replace('\d','')         #////Tweetlerin içindeki Sayıları temizler.
data['Tweet']= data['Tweet'].str.replace('gt', '')        #////Tweetlerin içindeki GT'leri temizler.
"""

"""
data.drop_duplicates(subset=["Tweet","KullaniciAdi","BegeniSayisi","Isim"], keep=False,inplace=False)   #///Tekrar eden satırları kaldırır
"""


data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))     #////Cümlelerin bütün harflerini küçültme
data['Isim'] = data['Isim'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))  #////İsimlerin bütün harflerini küçültme


data['Tweet'] = data['Tweet'].apply (lambda x: " ".join (x for x in x.split () if x not in sw))    #////Anlam ifade etmeyen kelimelerin çıkarılması
azgecenkelimeler = pd.Series(" ".join(data['Tweet']).split()).value_counts()                       #////Az geçen kelimeleri gösteriyor
silinecekkelimeler = pd.Series(" ".join(data['Tweet']).split()).value_counts()[-10:]               #////En az kullanılan son 10 kelimeyi gösterir
data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in silinecekkelimeler))   #////En az kullanılan son 10 kelimeyi siler.




#//// Sık geçen kelimelerin listelenmesi
"""
freq_df = data["Tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
freq_df.columns = ["kelimeler", "frekanslar"]
print(freq_df.head())

print("En sık geçen kelimelerin görselleştirilmesi : \n")
a = freq_df[freq_df.frekanslar > freq_df.frekanslar.mean() + freq_df.frekanslar.std()]
plt.bar(a["kelimeler"].head(5), a["frekanslar"].head(5))
plt.show()
plt.savefig("1-sık_geçen_kelimeler.pdf")
"""

#////Kelime Bulutunun Oluşturulması

Tweet = " ".join(i for i in data.Tweet)
print("Kelime bulutunun oluşturulması : \n")
wordcloud = WordCloud(background_color = "black").generate(Tweet)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


#data.to_excel ("tweetler_excel.xlsx")

