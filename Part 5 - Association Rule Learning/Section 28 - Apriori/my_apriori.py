# -*- coding: utf-8 -*-

# Apriori Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)

# This dataset is in the form of dataframe but for apriori we need list of list.
# List consist of several list where every list represent user i information

transaction=[]
for i in range(7501):
    transaction.append([str(dataset.values[i,j]) for j in range(20)])
    
from apyori import apriori
# min_support,min_confidence,min_left all are depends on the business model
rules=apriori(transaction,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
# min_length=2 represent that we get rules containg atleast two product

result=list(rules)
