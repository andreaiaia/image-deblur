PSNR_medie = open("PSNRpunto2.csv", 'w')

for i in range(75):
    PSNR_values = []
    for j in range(10):
        current = open(f"tests/sample{j+1}PSNR.csv", 'r')
        current_lines = current.readlines()
        current.close()
        PSNR_values.append(current_lines[i+1])
    
    noised = 0.0
    naive = 0.0
    regolarized = 0.0
    regolarized2 = 0.0
    totvar = 0.0

    for value in PSNR_values:
        fields = value.split(',')
        noised += float(fields[1])
        naive += float(fields[2])
        regolarized += float(fields[3])
        regolarized2 += float(fields[4])
        totvar += float(fields[5])

    PSNR_medie.write(f"{PSNR_values[0].split(',')[0]}, {noised/10}, {naive/10}, {regolarized/10}, {regolarized2/10}, {totvar/10}\n")
    
PSNR_medie.close()