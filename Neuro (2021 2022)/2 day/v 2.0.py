a = open("res.csv").read().replace("\n", ",").split(",")
f = []
for i in len(1,range(a)):
    if i%2==0:
        f.append(a[i])
fo = open("res1.txt", "w")
fo.write(",".join(f))