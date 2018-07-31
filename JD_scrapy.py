# /usr/bin/python
# -*- encoding:utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import time

jd_cookies = {'3AB9D23F7A4B3C9B':'T6STD3GR76F72F42HOLVYI5VOJP2CDKO5JJXPLPRCKS4Q5PIZ5FQCUNECRXV6CFNJK6WYHTVXEE5XTJV7VDRAA63LU',
              'PCSYCityID':'1',
              'TrackID':'1CbSlO9P-FCfWsd_q8PTyUCXKKuWowC0uMknyR3s0L3wsEVJX5olive8a5jtd-UcriJ_JNR2dLpArYOypsAvOow',
              '__guid':'181111935.2288895071492795400.1511613729280.6018',
              '__jda':'122270672.889453083.1477923798.1531639418.1533040235.195',
              '__jdb':'122270672.10.889453083|195.1533040235',
              '__jdc':'122270672',
              '__jdu':'889453083',
              '__jdv':'122270672|direct|-|none|-|1533040235115',
              '__utmz':'122270672.1520044941.2.2.utmcsr=trade.jd.com|utmccn=(referral)|utmcmd=referral|utmcct=/shopping/order/getOrderInfo.action',
              '_pst':'yaoleihxr',
              '_tp':'WnLgP0kBpakQ3SltjdAWbg%3D%3D',
              'areaId':'1',
              'ceshi3.com':'201',
              'cn':'0',
              'ipLoc-djd':'1-72-2799-0',
              'ipLocation':'%u5317%u4EAC',
              'monitor_count':'3',
              'pin':'yaoleihxr',
              'pinId':'aHCB4Y2o1YABNB2Km3viKQ',
              'shshshfp':'dd2277efcc6d2d49443a84012b005217',
              'shshshfpa':'b2728d0b-3488-0cac-be49-bfbe00b60abe-1531483949',
              'shshshfpb':'0e5a6161dd8a2344328e264bd67b84ae1aae57ca7959fa1745af43f92b',
              'shshshsID':'439e27bad69439e50a8cbd73eb0021a4_5_1533041067548',
              'thor':'DD49F6B28518B37B557FE849F3F2E1CDEB9862808A61DEF32151BF818C7B7384EDB6478A9BFD8945A95AF3F54492BBB7BBDB7D1E99D17BC7E84C24318ADF726FEF1C441400D531DF0BD28327B130F1DB6A261D7D767A39B5ABF5D94B00CE29D9D5710E2E461A1EA2B5775CE3A598CC6E383A324B1E852E0E7F899E56BEFA2CB1E513E0DDDB358D31480DFCED6B6A7D87',
              'unick':'yaoleihxr',
              'user-key':'0dd600fd-30b5-48e0-b123-7b7dd4cb691f',
              'wlfstk_smdl':'cf5vuah71mjeilp0ldex3iw9cy0bkt7x'}


url = 'https://sclub.jd.com/comment/productPageComments.action'
para = {'callback':'fetchJSON_comment98vv6366',
        'productId':'12377573',
        'score':'0',
        'sortType':'5',
        'page':'0',
        'pageSize':'10',
        'isShadowSku':'0',
        'fold':'1'}

def get_link(url):
    req = requests.get(url, cookies=jd_cookies)
    req.encoding = 'gbk'
    soup = BeautifulSoup(req.text, 'html.parser')
    print(soup)


if __name__ == '__main__':
    get_link('https://item.jd.com/12377573.html?dist=jd')
