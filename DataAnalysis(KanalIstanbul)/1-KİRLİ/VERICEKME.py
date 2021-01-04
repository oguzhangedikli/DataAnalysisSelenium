from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import csv
import time


def data():

    # Chromedriver yolunu gösterme ve açma
    driver = "C:\Program Files\Google\chromedriver.exe"
    tarayıcı = webdriver.Chrome(executable_path=driver)
    #


    # GİRİŞ YAPMA
    tarayıcı.get("https://twitter.com/login")
    time.sleep(1)
    tarayıcı.find_element_by_xpath(
        "//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[1]/label/div/div[2]/div/input").send_keys('data8889')
        tarayıcı.find_element_by_xpath("//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[2]/label/div/div[2]/div/input").send_keys('vericekme123')
    tarayıcı.find_element_by_xpath("//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/form/div/div[3]/div/div").click()
    time.sleep(2)
    #
    
    #ARAMA
    search = tarayıcı.find_element_by_xpath("//*[@id='react-root']/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div[2]/input")
    search.send_keys('kanal istanbul')
    time.sleep(2)
    search.send_keys(Keys.ENTER)
    time.sleep (2)
    tarayıcı.find_element_by_xpath ("//*[@id='react-root']/div/div/div[2]/main/div/div/div/div/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a/div").click ()
    time.sleep (2)
    #

    #CSV Oluşturma
    file = open("data.csv", "w", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["Tweet"])
    #


    #SCRULL
    a = 0
    while a < 75:
        lastHeight = tarayıcı.execute_script("return document.body.scrollHeight")
        i = 0
        while i < 1:
            tarayıcı.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            newHeight = tarayıcı.execute_script("return document.body.scrollHeight")
            if newHeight == lastHeight:
                break
            else:
                lastHeight = newHeight

            i = i + 1

        page_source = tarayıcı.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        tweets = soup.find_all("div", attrs={"data-testid": "tweet"})
        print(tweets)
        for tweet in tweets:
            try:
                Tweet = tweet.find("div", attrs={
                    'css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0'}).text
                writer.writerow([Tweet])

            except:
                print("**")

        a = a + 1
    #

data()

#DIŞA AKTARMA
from pandas import read_csv

kayıt = read_csv("data.csv")
kayıt.to_excel("data_excel.xlsx")
