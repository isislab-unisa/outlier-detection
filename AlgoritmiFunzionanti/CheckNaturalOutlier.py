import pandas as pd
import numpy as np


def checkDbpediaDates():

    dfSpqrqlBirthDate = pd.read_csv('sparqlBirthdate', error_bad_lines=False,nrows=37)
    dfdbpediawikiData = pd.read_csv('dbPedia&WikidataBirthDate', error_bad_lines=False)
    for i, row in dfSpqrqlBirthDate.iterrows():
        row['Concept'] = str(row['Concept'])

    for i, row in dfSpqrqlBirthDate.iterrows():
        row['Concept'] = row['Concept'].replace("-", "")

    dfSpqrqlBirthDate['ValoriNumerici'] = dfSpqrqlBirthDate['Concept'].astype(np.int64)
    for j, row2 in dfdbpediawikiData.iterrows():
        row2['d'] = str(row2['d'])

    for j, row2 in dfdbpediawikiData.iterrows():
        row2['d'] = row2['d'].replace("-", "")

    dfdbpediawikiData['ValoriNumerici'] = dfdbpediawikiData['d'].astype(np.int64)
    a=np.zeros(shape=(len(dfSpqrqlBirthDate),1))
    dfSpqrqlBirthDate['Corrispondenza'] = a.astype(np.int64)
    for k,row3 in dfSpqrqlBirthDate.iterrows():
        for l,row4 in dfdbpediawikiData.iterrows():
            if row3['ValoriNumerici'] == row4['ValoriNumerici']:
                dfSpqrqlBirthDate['Corrispondenza'][k]=1
                print(row3['ValoriNumerici'])
                break



    return dfSpqrqlBirthDate,dfdbpediawikiData




