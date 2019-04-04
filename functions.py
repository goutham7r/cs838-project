import urllib2
from datetime import datetime
import bs4  as BS
import re


# name parameter has to be of the form first_name_last_name

def calcAge(name):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    prefix='http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvsection=0&titles='
    suffix='&format=xml'
    infile = opener.open(prefix+name+suffix)
    page2 = infile.read()
    soup = BS.BeautifulSoup(page2)
    #print soup.prettify()
    birth_re = re.search(r'(Birth date(.*?)}})', soup.getText())
    birth_data = birth_re.group(0).split('|')
    year = int(birth_data[2])
    today = datetime.today()
    age=today.year-year
    return age

def calcGender(name):
    try:
        name=name.split('_')
        urlname=name[0]
        for i in range(1,len(name)):
            urlname=urlname+'+'+name[i]
    except:
        print 'Name not in correct format'
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    prefix='https://www.imdb.com/search/name?name='
    suffix='&gender=male'
    url=prefix+urlname+suffix
    infile = opener.open(url)
    page2 = infile.read()
    soup = BS.BeautifulSoup(page2)
    birth_re = re.search(r'No results', soup.getText())

    if birth_re is None:
        gender='male'
    else:
        gender='female'
    return gender    
