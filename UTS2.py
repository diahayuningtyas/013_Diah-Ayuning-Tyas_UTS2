#Diah Ayuning Tyas_A_21091397013
#Membuat Program Multi Neuron Batch Input

# inisialisasi numpy
import numpy as np

#inisialisasi variabel matriks 6x10 (jumlah batch = 6 dan input layer = 10 )
inputs = [[1.2,2.5,3.0,1.0,2.3,1.5,5.0,8.0,3.5,6.0],
          [1.2,4.2,5.5,3.5,8.0,6.5,2.5,1.0,4.5,5.0],
          [2.0,5.0,2.5,1.5,4.0,3.5,5.5,9.0,6.0,7.0],
          [3.3,5.0,2.4,1.2,5.0,3.2,7.5,6.0,3.0,2.5],
          [1.2,2.5,3.5,7.0,2.0,1.5,5.0,5.5,8.0,6.5],
          [3.5,2.5,1.0,2.2,4.0,2.0,8.0,5.0,3.5,7.5]]

#inisialisasi bobot variabel layer 1 (panjang weight = panjang input dan jumlah weight = jumlah neuron)
weights1 = [[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
           [1.2,2.2,4.2,3.4,5.0,1.0,2.5,5.2,4.7,8.0],
           [4.0,2.5,2.2,3.5,1.5,6.0,2.3,1.5,8.0,9.0],
           [3.0,5.5,2.5,4.2,1.0,2.6,7.5,9.0,2.0,6.7],
           [1.2,3.0,3.2,4.5,7.0,6.5,3.0,2.5,8.5,6.5]]

#inisialisasi bias layer 1 (jumlah bias = jumlah neuron)
bias1 = [1.2,2.0,2.5,3.2,3.0]

#inisialisasi bobot variabel layer 2 (panjang weight = panjang input dan jumlah weight = jumlah neuron)
weights2 = [[5.5,0.8,-1.2,3.5,4.0],
            [3.0,6.0,0.7,-2.4,1.0],
            [2.0,1.5,2.0,-0.5,0.2]]

#inisialisasi bias layer 2
bias2 = [1.2,0.4,5.0] 

#Menghitung output menggunakan rumus : 
output = np.dot(inputs, np.array(weights1).T) + bias1
output2 = np.dot(output, np.array(weights2).T) + bias2

#Mencetak Output 
print(output2)