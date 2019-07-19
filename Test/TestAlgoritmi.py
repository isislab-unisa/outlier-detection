import unittest
from AlgoritmiFunzionanti import LofExample as lf, IQRMethod
from AlgoritmiFunzionanti import ZScore as zs
from pandas import DataFrame
from AlgoritmiFunzionanti import ModifiedZScore
from AlgoritmiFunzionanti import DBScan
import pandas as pd

Cars = {
    'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'l'],
    'Price': [22, 2500, 27, 35, 34, 23, 78, 65, -4600, 78, 19, 28, 12, 18]
}
df = DataFrame(Cars, columns=['Brand', 'Price'])

dfBellezza = pd.read_csv('datasetBellezza.csv', error_bad_lines=False, sep=';')

dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')

dfMappe = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv',error_bad_lines=False,sep=";",encoding="ISO-8859-1")
class TestLOF(unittest.TestCase) :





    def testDBFittizzio(self):
        mylist = list()
        mylist.append(1)
        mylist.append(8)
        self.assertListEqual(zs.outliers_z_score(df["Price"], df, 1.5), mylist)  # ZScore
        self.assertListEqual(lf.DBFittizzio(10, df), mylist)  #LOF
    #self.assertListEqual(IQRMethod.outliers_iqr(df["Price"], df), mylist)  #IQR
        self.assertListEqual(ModifiedZScore.outliers_modified_z_score(df["Price"], df, 4),mylist)
        self.assertListEqual(DBScan.dbSCan(0.3, df, 1), mylist)


    def testDBMappe(self):
        myList=list()
        myList.append(154)
        myList.append(1671)
        self.assertListEqual(zs.outliers_z_score(dfMappe["Longitudine"], dfMappe, 3), myList)  # ZScore
        self.assertListEqual(lf.dbMappe(0.0007), myList)  # LOF
        self.assertListEqual(ModifiedZScore.outliers_modified_z_score(dfMappe["Longitudine"], dfMappe, 3), myList)
        self.assertListEqual(DBScan.dbSCan(1.8530333428933457e-06, dfMappe, 4), myList)


    def testVariazioniDelibere(self):
        myList = [1, 3, 5]
        self.assertListEqual(zs.outliers_z_score(dfVariazioni["Unita'OrganizzativaDirigenziale"],dfVariazioni,3),myList)
        self.assertListEqual(lf.dbDatasetVariazioniDelibere(0.0026), myList)  # LOF
        self.assertListEqual(ModifiedZScore.outliers_modified_z_score(dfVariazioni["Unita'OrganizzativaDirigenziale"],dfVariazioni,70000),myList)
        self.assertListEqual(DBScan.dbSCan(0.00043327556326473135, dfVariazioni, 6), myList)



    def testDatasetBellezza(self):
        myList=[0,1,6]
        #self.assertListEqual(zs.outliers_z_score(dfBellezza["Importo"], dfBellezza, 6), myList)
        #self.assertListEqual(lf.dbDatasetVariazioniDelibere(0.0026), myList)  # LOF
        #self.assertListEqual(ModifiedZScore.outliers_modified_z_score(dfBellezza["Importo"], dfBellezza,70000), myList)
        #self.assertListEqual(DBScan.dbSCan(0.00043327556326473135, dfBellezza, 6), myList)'''


