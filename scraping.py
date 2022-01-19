MSE_medie = open("MSEmedie.csv", 'w')

for i in range(8):
    MSE_values = []
    for j in range(10):
        current = open(f"scraped/sample{j+1}MSE.csv", 'r')
        current_lines = current.readlines()
        current.close()
        MSE_values.append(current_lines[i+1])
    
    noised = 0.0
    naive = 0.0
    regolarized = 0.0
    regolarized2 = 0.0
    totvar = 0.0

    for value in MSE_values:
        fields = value.split(',')
        #output = open(f"scraped/medie/MSE-{fields[0]}.csv", 'a')
        noised += float(fields[1])
        naive += float(fields[2])
        regolarized += float(fields[3])
        regolarized2 += float(fields[4])
        totvar += float(fields[5])

    MSE_medie.write(f"{MSE_values[0].split(',')[0]}, {noised/10}, {naive/10}, {regolarized/10}, {regolarized2/10}, {totvar/10}\n")
    
MSE_medie.close()