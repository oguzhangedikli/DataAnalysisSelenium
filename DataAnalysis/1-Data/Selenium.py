from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv
import pandas as pd
driver_path = "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
browser = webdriver.Chrome(executable_path=driver_path)
browser.get("https://www.google.com/")
