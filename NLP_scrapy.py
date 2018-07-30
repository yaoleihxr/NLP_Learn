# /usr/bin/python
# -*- encoding:utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re


db_cookies = {'__utma':'30149280.1643786549.1477397830.1532749880.1532766953.36',
              '__utmb':'30149280.1.10.1532766953',
              '__utmc':'30149280',
              '__utmt':'1',
              '__utmv':'30149280.16',
              '__utmz':'30149280.1532766953.36.33.utmcsr=blog.csdn.net|utmccn=(referral)|utmcmd=referral|utmcct=/c091728/article/details/78347915',
              '_ga':'GA1.2.1643786549.1477397830',
              '_gid':'GA1.2.2131697169.1532767633',
              '_vwo_uuid_v2':'ABD20203674CEC5936B3DF4EE04A2BB2|ea2a27a31fc4b6933df9ba3bf9cca346',
              'ap':'1',
              'bid':'OmP2I49YLk0',
              'ck':'_a9U',
              'dbcl2':'160585238:aTEflnvSZfg',
              'gr_user_id':'87ff83c8-1f1f-4f14-9a88-fa3f0a16d62e',
              'll':'108288',
              'ps':'y',
              'push_doumail_num':'0',
              'push_noty_num':'0',
              'viewed':'2061116_26919485_26698660_25804112'}

num_per_page = 20

def get_link(url):
    req = requests.get(url, cookies=db_cookies)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'html.parser')
    print(soup)
    list_votes = soup.find_all('span',{'class':'votes'})
    list_comment = soup.find_all('span', {'class':'comment-info'})
    list_short = soup.find_all('span', {'class':'short'})
    comment_list = []
    for i in range(num_per_page):
        print(list_comment[i])
        cm_id = list_comment[i].find_all('a')[0].text
        cm_star = list_comment[i].find_all('span', {'class':re.compile(r'^allstar.*')})[0]['title']
        cm_time = list_comment[i].find_all('span', {'class':'comment-time '})[0]['title']
        cm_short = list_short[i].text
        cm_votes = list_votes[i].text
        comment_list.append([cm_id, cm_star, cm_time, cm_short, cm_votes])
    print(len(comment_list))


if __name__ == '__main__':
    get_link('https://movie.douban.com/subject/26608228/comments?start=0&limit=20')
