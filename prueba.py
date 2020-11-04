# A = [1,2,3,4,5,6,7,8,9]
# B = ["A","B","C"]
# for a in zip (A[1:],A[:]):
# 	print(a)

str_ = "../data/laser/04-Nov-2020/exp1_5401q851dfas.asc"
print(str_.rfind("/"))
indx=str_.rfind("/")
print(str_[:indx]+"/events"+str_[indx:-4]+"control_events.asc")