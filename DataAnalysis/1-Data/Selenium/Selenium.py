import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def data():
    kullanici = input("Kullanıcı Adını Giriniz: ")
    sifre = input('Şifrenizi Giriniz: ')
    sayfa = int(input("Çekmek istediğiniz Sayfa sayısını giriniz = "))
    kelime = input("Aramak İstediğiniz Kelimeyi Giriniz: ")
    driver_path = "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
    browser = webdriver.Chrome(executable_path=driver_path)
    browser.get("https://chrome.google.com/webstore/detail/adblock-plus-free-ad-bloc/cfhdojbkjhnklbpkdaibdccddilifddb?hl=tr")
    browser.set_window_size(1920, 1080)
    time.sleep(15)
    browser.get("https://twitter.com/login")
    time.sleep(1)
    browser.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[1]/label/div/div[2]/div/input").send_keys(
        kullanici)
    time.sleep(1)
    browser.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[2]/label/div/div[2]/div/input").send_keys(
        sifre)
    time.sleep(1)
    browser.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[3]/div/div").click()
    time.sleep(5)
    aranacak = browser.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div[2]/input")
    aranacak.send_keys(kelime)
    time.sleep(2)
    aranacak.send_keys(Keys.ENTER)
    time.sleep(2)
    browser.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div/div/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a/div").click()
    time.sleep(2)
    # browser.get('https://twitter.com/search?q=asgari%20%C3%BCcret&src=typed_query')
    file = open("tweetler.csv", "w", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["Isim", 'KullaniciAdi', "YorumSayisi", "RetweetSayisi", "BegeniSayisi", "Tweet"])

    a = 0
    while a < sayfa:
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        i = 0
        while i < 1:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            newHeight = browser.execute_script("return document.body.scrollHeight")
            if newHeight == lastHeight:
                break
            else:
                lastHeight = newHeight

            i = i + 1

        sayfakaynagi = browser.page_source
        soup = BeautifulSoup(sayfakaynagi, "html.parser")
        tweetler = soup.find_all("div", attrs={"data-testid": "tweet"})
        print(tweetler)
        for tweet in tweetler:
            try:
                Isim = tweet.find("div", attrs={
                    'css-901oao css-bfa6kz r-jwli3a r-1qd0xha r-a023e6 r-b88u0q r-ad9z0x r-bcqeeo r-3s2u2q r-qvutc0'}).text
                KullaniciAdi = tweet.find("div", attrs={
                    'css-901oao css-bfa6kz r-111h2gw r-18u37iz r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-qvutc0'}).text
                Tweet = tweet.find("div", attrs={
                    'css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0'}).text
                YorumSayisi = tweet.find("div", attrs={"data-testid": "reply"}).text
                RetweetSayisi = tweet.find("div", attrs={"data-testid": "retweet"}).text
                BegeniSayisi = tweet.find("div", attrs={"data-testid": "like"}).text
                writer.writerow([Isim, KullaniciAdi, YorumSayisi, RetweetSayisi, BegeniSayisi, Tweet])

            except:
                print("**")

        a = a + 1


data()

from pandas import read_csv

ss = read_csv("tweetler.csv")
ss.to_excel("tweetler_excel.xlsx")
