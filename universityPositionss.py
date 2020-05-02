import pymongo
import requests
from bs4 import BeautifulSoup
import bs4
#通过url获取页面信息
def getHTMLText():
    try:
        url = 'http://www.zuihaodaxue.com/zuihaodaxuepaiming2019.html'
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        print('网络连接失败')
        return ""
#提取所需要的大学排名信息
def fillList(demo):
    soup = BeautifulSoup(demo,'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds=tr('td')
            fllist.append([tds[0].string,tds[1].string,tds[3].string])
#对大学排名进行显示输出
def PrintFllist(fllist,num):
    # 建立连接
    client = pymongo.MongoClient('localhost', 27017)
    # 连接数据库
    db = client['universityPosition']
    # 连接表
    collection = db['rank']
    for i in range(num):
        u = fllist[i]
        # 插入数据
        collection.insert_one({'排名': u[0], '学校名称': u[1],'总分':u[2]})
if __name__=='__main__':
    fllist= []
    demo=getHTMLText()
    fillList(demo)
    PrintFllist(fllist,20)  #20 所
