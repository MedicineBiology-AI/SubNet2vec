file1=open("drug_disease_prediction.txt","w")
for lines in open("drug_disease_proximity.txt","r"):
    line=lines.strip().split("\t")
    # print(line[0])
    # print(line[1])
    # print(line[2])
    # print(line[4])
    line[4]=float(line[4])
    if line[4]<=-0.15:
        line[4]=1
    else:
        line[4]=0
    result="%s\t%s\t%s\n"%(line[2],line[0],line[4])
    file1.write(result)
file1.close()





