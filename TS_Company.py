# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:43:07 2020

@author: tejas
"""

import pandas as pd
import numpy as np
company=pd.read_csv("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\DECISION TREE\Company\Company_Data.csv")
company.reset_index()
company.isnull().sum()
company.Sales.mean() ########Here we got the mean of 7.5
company.head()
company["sales"]="<=7.5"
company.loc[company["Sales"]>=7.5,"sales"]="High Sales"
company.loc[company["Sales"]<=7.5,"sales"]="Low Sales"
company=company.drop(["Sales"],axis=1)
company.ShelveLoc.unique()
colnames=list(company.columns)
predictors=colnames[:10]
target=colnames[10]
###Label Encoders#########
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
select_columns=["ShelveLoc","Urban","US","sales"]
le.fit(company[select_columns].values.flatten())
company[select_columns]=company[select_columns].apply(le.fit_transform)

##split the data into train and testing###############
from sklearn.model_selection import train_test_split
train,test=train_test_split(company,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion="entropy")
train_model=model.fit(train[predictors],train[target])
train_pred=train_model.predict(train[predictors])

#####Accuracy##########
train_accu=np.mean(train.sales==model.predict(train[predictors]))########100%

######Crosstab#########
train_crosstab=pd.crosstab(train[target],train_pred)

####test #############
test_model=model.fit(test[predictors],test[target])
test_pred=test_model.predict(test[predictors])
#######Accuarcy########
test_accu=np.mean(test.sales==model.predict(test[predictors]))#######100%
###crosstab############
test_crosstab=pd.crosstab(test[target],test_pred)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

colnames = list(company.columns)
predictors = colnames[:10]
target = colnames[10]

dot_data = StringIO()

export_graphviz(model,out_file = dot_data, filled =True, rounded = True, feature_names =predictors,class_names = target, impurity = False )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


##Creating the pdf file of decision tree
graph.write_pdf("company.pdf")
pwd

##Creating a png file of the decsion tree
graph.write_png('company.png')
#
