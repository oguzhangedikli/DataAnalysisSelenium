import pandas as pd

data = pd.read_excel("Category_Tweet.xlsx")

#//////Kategorisi olmayan verileri siliyor
#df = data.dropna(axis=0, how = 'any')
#


#//////Ana dosyadan kategorisi olan verileri siliyor
#df = data.drop(index=['Ekonomi','Siyasi','Toplumsal'], axis=0)
#

df.to_excel('DataTweet.xlsx')