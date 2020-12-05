import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob

#test opening one txt file
review = pd.read_csv('/Users/melissakozak/Desktop/deep_learning_with_python/data/aclImdb/train/neg/0_3.txt', delimiter="\t", header = None)
review

my_dir = '/Users/melissakozak/Desktop/deep_learning_with_python/data/aclImdb/train/neg/'
filelist = []
filesList = []
os.chdir ( my_dir )

for f in glob.glob("*.txt"):
    fileName, fileExtension = os.path.splitext(f)
    filelist.append(fileName)
    filesList.append(f)

df = pd.DataFrame()
for a in filesList:
    frame = pd.read_csv(a, delimiter="\t", header = None)
    df = df.append(frame)

df.head()
df.info()
df.shape
df.columns
df = df.drop([1, 2, 3, 4, 5, 6, 7], axis=1)


df.info()
