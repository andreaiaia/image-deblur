import numpy as np

MSE_medie = open("MSEstd.csv", 'w')

for i in range(8):
    MSE_values = []
    for j in range(10):
        current = open(f"scraped/sample{j+1}MSE.csv", 'r')
        current_lines = current.readlines()
        current.close()
        MSE_values.append(current_lines[i+1])
    
    noised = []
    naive = []
    regolarized = []
    regolarized2 = []
    totvar = []

    for value in MSE_values:
        fields = value.split(',')
        noised.append(float(fields[1]))
        naive.append(float(fields[2]))
        regolarized.append(float(fields[3]))
        regolarized2.append(float(fields[4]))
        totvar.append(float(fields[5]))

    MSE_medie.write(f"{MSE_values[0].split(',')[0]}, {np.std(noised)}, {np.std(naive)}, {np.std(regolarized)}, {np.std(regolarized2)}, {np.std(totvar)}\n")
    
MSE_medie.close()