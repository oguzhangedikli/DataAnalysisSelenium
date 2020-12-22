from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv
import pandas as pd

def data():
    sayfa = int(input("Çekmek istediğiniz Sayfa sayısını giriniz = "))
    driver_path = "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
    browser = webdriver.Chrome(executable_path=driver_path)
    browser.get("https://twitter.com/search?q=Asgari%20%C3%BCcret&src=typed_query&f=live")
    
    file = open("tweetler.csv","w",encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["Isim","KullanıcıAdi","YorumSayisi","RetweetSayisi","BegeniSayisi","Tweet"])
    
    a = 0
    while a < sayfa:
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        i=0
        while i<1:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            newHeight = browser.execute_script("return document.body.scrollHeight")
            if newHeight == lastHeight:
                break
            else:
                lastHeight = newHeight

            i = i+1

        sayfakaynagi = browser.page_source
        soup = BeautifulSoup(sayfakaynagi, "html.parser")
        tweetler = soup.find_all ("div", attrs={"data-testid":"tweet"})
        print (tweetler)
        for tweet in tweetler:

            try:
                Isim = tweet.find("div", attrs={"css-901oao css-bfa6kz r-18jsvk2 r-1qd0xha r-a023e6 r-b88u0q r-ad9z0x r-bcqeeo r-3s2u2q r-qvutc0"}).text
                KullanıcıAdı = tweet.find("div", attrs={"css-901oao css-bfa6kz r-m0bqgq r-18u37iz r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-qvutc0"}).text
                Tweet = tweet.find("div", attrs={"css-901oao r-18jsvk2 r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0"}).text
                YorumSayisi = tweet.find("div", attrs={"data-testid":"reply"}).text
                RetweetSayisi = tweet.find("div", attrs={"data-testid":"retweet"}).text
                BegeniSayisi = tweet.find("div", attrs={"data-testid":"like"}).text
                writer.writerow([Isim,KullanıcıAdı,YorumSayisi,RetweetSayisi,BegeniSayisi,Tweet])

            except:
                print("**")
            
        a = a+1
data()

import pandas as pd
ss = pd.read_csv("tweetler.csv")
ss.to_excel("tweetler_excel.xlsx")
