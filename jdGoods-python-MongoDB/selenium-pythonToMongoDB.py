import time
import pymongo
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from lxml import etree

# 此处可以不打开浏览器进行爬取
# options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# browser = webdriver.Chrome(options=options)

# 为了演示方便,打开谷歌浏览器进行爬取
browser = webdriver.Chrome()
# 显式等待
wait = WebDriverWait(browser, 50)


def search(url):
    # 该方法用于模拟打开jd，输入内容，模拟点击。返回总页数
    browser.get(url)
    # 1. 获取输入框，输入'键盘'
    input = wait.until(
        # XPATH可以在网页上获取
        EC.presence_of_element_located((By.XPATH, '//*[@id="key"]'))
    )
    input.clear()
    input.send_keys('键盘')
    # 2. 获取搜索按钮，实现点击
    button = wait.until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="search"]/div/div[2]/button'))
    )
    button.click()
    # 3. 执行js，拖动窗口,如果不往下拖,京东商品不会显示出来(属于懒加载)
    # 为了防止操作过快,页面还未加载出来,先设置睡眠一秒
    time.sleep(1)
    for i in range(16):
        js = 'window.scrollTo(0, {} * document.body.scrollHeight / 16)'.format(i)
        browser.execute_script(js)
        time.sleep(0.5)

    # 4. 获取搜素到的关键词查询出的总页数
    # //*[@id="J_bottomPage"]/span[2]/em[1]/b
    total = wait.until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="J_bottomPage"]/span[2]/em[1]/b'))
    )
    # TODO: 测试源码中是否有60个商品信息
    html = browser.page_source
    data = parse_html(html)
    print(data)
    print(len(data))
    save_mongo(data)

    return total.text


def next_page():
    # 获取下一页的内容
    next = wait.until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="J_bottomPage"]/span[1]/a[9]'))
    )
    next.click()
    # 滚动
    for i in range(16):
        js = 'window.scrollTo(0, {} * document.body.scrollHeight / 16)'.format(i)
        browser.execute_script(js)
        time.sleep(0.5)

    time.sleep(2)
    html = browser.page_source
    # print(html)
    data = parse_html(html)
    print(data)
    print(len(data))
    save_mongo(data)


def parse_html(html):
    tree = etree.HTML(html)
    goods_list = tree.xpath('//*[@id="J_goodsList"]/ul/li')
    result = []
    for goods in goods_list:
        # TODO: 获取图片、价格、名称、评论、店的名称
        data = {
            'name': goods.xpath('./div/div[3]/a/em/text()'),
            'price': goods.xpath('./div/div[2]/strong/i/text()'),
            # 如果存在懒加载的话,图片信息会存在于img的data-lazy-img属性中,此处进行判断
            'img': goods.xpath('./div/div[1]/a/img/@src') if goods.xpath('./div/div[1]/a/img/@src') else goods.xpath(
                './div/div[1]/a/img/@data-lazy-img'),
            'shop': goods.xpath('./div/div[5]/span/a/text()')
        }
        result.append(data)
    return result


# 创建连接mongobd数据库并写入函数
def save_mongo(data):
    client = pymongo.MongoClient(host='127.0.0.1', port=27017)
    db = client['jd']
    db['键盘'].insert_many(data)


def main():
    url = 'http://www.jd.com'
    total = search(url)
    # 此处可设置需要爬取多少页面的数据
    # 例如爬取搜素到的关键词查询出的总页数得数据
    # for page in range(2, int(total)+1)
    for page in range(2, 4):
        next_page()


if __name__ == '__main__':
    main()
