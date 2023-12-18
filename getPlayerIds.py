import re
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from bs4 import BeautifulSoup
import csv
soup = BeautifulSoup(open("India.html", encoding="utf8"), "html.parser")


# Find all the <a> elements with 'data-link' class that contain "/ci/content/player/"
player_elements = soup.find_all('a', class_='data-link', href=lambda href: href and "/ci/content/player/" in href)

# Create a dictionary to store the player IDs and names
India_player_dict = {}

# Iterate through the player elements and extract the player ID and name
for player_element in player_elements:
    href = player_element['href']
    player_id = href.split('/ci/content/player/')[1].split('.html')[0]
    player_name = player_element.text
    India_player_dict[player_id] = player_name

print(India_player_dict)

with open('IndiaSquad.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in India_player_dict.items():
       writer.writerow([key, value])