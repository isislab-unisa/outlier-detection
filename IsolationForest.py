from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
df = pd.read_csv('sparqlBirthDate',error_bad_lines=False)

for i,row in  df.iterrows():
        row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
        row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)

X= df.iloc[:,1].values.reshape(-1,1)


clf = IsolationForest( )
preds = clf.fit_predict(X)
print(preds)
