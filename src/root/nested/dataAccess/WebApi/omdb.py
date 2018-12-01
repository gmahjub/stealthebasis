from urllib.request import urlretrieve, urlopen, Request
from bs4 import BeautifulSoup

import pandas as pd
import requests
import json

class omdb(object):

    """description of class"""

    def __init__(self, **kwargs):
        print("ombdb_init")
        self.omdbp_api_key = "87555ca"
        return super().__init__(**kwargs)

    def get_flatfile_from_web(self, 
                              url, 
                              filename,
                              sep = ','):

        # returns a dataframe of the downloaded file, assumes the flat file is in .csv format
        # if not csv, identify seperator with sep

        # datacamp url example
        # https://s3.amazonaws.com/assets.datacamp.com/production/course_1066/datasets/winequality-red.csv

        # filename should be a full complete path to a local file
        urlretrieve(url = url, filename = filename)
        df = pd.read_csv(filename, sep = sep )
        return (df)

    def make_urllibRequest_get_resp_html(self,
                                         url):

        request = Request(url)
        response = urlopen(request)
        html = response.read()
        response.close()
        return (html)

    def make_request_get_text(self,
                              url):

        r = requests.get(url)
        text = r.text
        return(text)

    def make_request_get_pretty_html(self,
                                     url):

        r = requests.get(url)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, features = "html5lib")
        pretty_soup = soup.prettify()
        return (pretty_soup)

    def get_html_title_from_soup_obj(self,
                                     soup_obj):

        return(soup_obj.title)

    def get_text_from_soup_obj(self,
                               soup_obj):

        return (soup_obj.text)

    def get_hyperlinks_from_soup_obj(self,
                                     soup_obj):

        a_tags = soup_obj.find_all('a')
        for link in a_tags:
            print (link.get('hfref'))
        return (a_tags)



omdb_obj = omdb()
omdb_obj.make_request_get_pretty_html("https://www.python.org/~guido")