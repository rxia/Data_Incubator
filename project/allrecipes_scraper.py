from recipe_scrapers import scrape_me
import pickle
import numpy as np
import time

data_allrecipes = []

for ID in np.arange(129001,299999):
    try:
        scrape_result = scrape_me('http://allrecipes.com/Recipe/{}'.format(ID))
        recipe_i = {}
        recipe_i['id'] = ID
        recipe_i['title'] = scrape_result.title()
        recipe_i['total_time'] = scrape_result.total_time()
        recipe_i['ingredients'] = scrape_result.ingredients()
        recipe_i['instruction'] = scrape_result.instructions()
        recipe_i['links'] = scrape_result.links()
        data_allrecipes.append(recipe_i)
    except:
        print('ID {} is not valid'.format(ID))
    if ID%100==0:
        print('at ID {} data length = {}'.format(ID,len(data_allrecipes)))
    if ID%1000==0:
        print('file saved at {}'.format(ID))
        with open('project/data/scraped_data_allrecipes.pickle', 'wb') as f:
            pickle.dump(data_allrecipes, f)


##
def getRecipeLinks(id):
    page = requests.get('http://allrecipes.com/recipe/' + str(id))
    tree = html.fromstring(page.text)

    # I want to get the text in the src="text" in order to get the imagesource url.
    imageSrcURL = tree.xpath('//img[@class="rec-photo"]/src')
    # gets the file from the source url from allrecipes website
    file = cStringIO.StringIO(urllib.urlopen(imageSrcURL).read())
    # gets the image data
    img = Image.open(file)

