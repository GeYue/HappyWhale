#coding=UTF-8

import csv
import pandas as pd 

sub_files = [
	'./historized_submissions/submission-v9.csv',
	'./historized_submissions/submission-v19.csv',
	'./historized_submissions/submission-v23.csv',
]

# Weights of the individual subs
sub_weight = [
    0.720**5,
    0.745**2,
    0.734**3,
]

Hlabel = 'image' 
Htarget = 'predictions'
npt = 6
place_weights = {}
for i in range(npt):
    place_weights[i] = ( 1 / (i + 1) )
    
print(place_weights)

lg = len(sub_files)
sub = [None]*lg

for i, file in enumerate( sub_files ):   
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

out = open("./historized_submissions/submission.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()
