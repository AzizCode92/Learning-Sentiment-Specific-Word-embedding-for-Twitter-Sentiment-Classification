import csv
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

from random import randint


inputFileName = "/root/PycharmProjects/project_text_mining/data/trained.csv"

with open(inputFileName) as csvfile :
    readCSV = csv.reader(csvfile, delimiter=',')




    for row in readCSV:


        if (row[0]=="0"):
            row[0]= "negative"
        elif (row[0]=="4"):
            row[0]= "positive"
        else : row [0]=="neutral"

        t1 = row[4]
        t2 = row[0]

        row[4] = row [0]
        row[0]=t1
        row[0]=row[1]
        row[1]=t1
        #print(row[1])
        del(row[2])
        del(row[2])
        row[1]= str(randint(11111111, 99999999))

        print '\t'.join(str(p) for p in row)
        #print "\t".join(row)


