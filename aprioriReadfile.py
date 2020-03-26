# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:20:08 2020

@author: Administrator
"""

from apyori import apriori

transactions=[]
f = open('apriori22.txt','r',encoding='utf-8')

lines = f.readlines()
   
for line in lines:
    transactions.append(line.strip().split(','))

f.close()

res=list(apriori(transactions,min_support=0.5,min_confidence=0.8))

for i in res:
    print(i)
    
