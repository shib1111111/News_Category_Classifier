from bs4 import BeautifulSoup
import requests

def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headline = soup.find('h1').get_text()
    description = soup.find_all('p')
    description_text = ''.join([tag.get_text() for tag in description])
    if len(description_text) == 0 :
      description_text = headline
    return headline, description_text

#headline, description = scrape_content('https://www.moneycontrol.com/news/business/real-estate/home-and-dry-5-mumbai-areas-that-see-waterlogging-every-year-during-monsoon-10868831.html')
#print(headline)
#print(description)