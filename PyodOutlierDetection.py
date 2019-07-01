from pandas import  DataFrame
from AlgoritmiFunzionanti import lof
Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,2500,27,35,34,23,78,65,-4600,78,19,28,12,18]
        }
df = DataFrame(Cars, columns=['Brand', 'Price'])

X=[df['Price'].tolist(),df['Price'].tolist()]

test_lof = lof.LOF(X, lof.LOF.CONST_EUCLIDEAN, 7)
lofs = test_lof.get_lof_sorted_filtered(True)
df['lofs']=lofs
for l in lofs:
    print(str(l[0]) + ": " + str(l[1]))
