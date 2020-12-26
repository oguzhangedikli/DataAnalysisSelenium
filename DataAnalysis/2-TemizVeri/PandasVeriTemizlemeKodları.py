import pandas as pd
import numpy as np
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


data = pd.read_excel ("KullanılanVeriler/KirliVeri.xlsx" )

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

"""
data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))     #////Cümlelerin bütün harflerini küçültme
data['Isim'] = data['Isim'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))  #////Cümlelerin bütün harflerini küçültme
"""

"""
data['Tweet'] = data['Tweet'].apply (lambda x: " ".join (x for x in x.split () if x not in sw))    #////Anlam ifade etmeyen kelimelerin çıkarılması
azgecenkelimeler = pd.Series(" ".join(data['Tweet']).split()).value_counts()                       #////Kelimelerin kaç defa geçtiğini gösteriyor
silinecekkelimeler = pd.Series(" ".join(data['Tweet']).split()).value_counts()[-10:]               #////En az kullanılan son 10 kelimeyi gösterir
data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in silinecekkelimeler))   #////En az kullanılan son 10 kelimeyi siler.
"""

"""
kelimesayisi = data["Tweet"].apply (lambda x: pd.value_counts (x.split (" "))).sum (axis=0).reset_index ()
kelimesayisi.columns = ["kelimeler", "sayıları"]
print(kelimesayisi.head())                                                                #////Kelimelerin kaç kez geçtiğini listeler.
"""

"""

#////Kelime Bulutunun Oluşturulması

Tweet = " ".join(i for i in data.Tweet)
print("Kelime bulutunun oluşturulması : \n")
wordcloud = WordCloud(background_color = "white").generate(Tweet)
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
"""

#data.to_excel ("tweetler_excel.xlsx")

