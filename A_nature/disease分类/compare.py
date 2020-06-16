pos_pairs=[]
for lines in open("postive_drug_disease.txt"):
    line=lines.strip().split("\t")
    pos_pairs.append([line[0],line[1]])
print(len(pos_pairs))
pre_pairs=[]
for lines in open("predict_top100_drug_disease.txt"):
    line=lines.strip().split("\t")
    pre_pairs.append([line[0],line[1]])
print(len(pre_pairs))
count=0
for pre in pre_pairs:
    if pre in pos_pairs:
        print(pre)
        count=count+1
print(count)

