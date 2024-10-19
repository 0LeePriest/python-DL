import matplotlib.pyplot as plt
#Matlab方式
Plt1 = plt.figure()
x= [1,2,4,5,6]
y= [3,5,8,9,10]
plt.plot(x,y)
plt.show()
#面向对象的方式
plt2 = plt.figure()
x= [1,2,3,4,5]
y= [1,2,3,4,5]
Pla = plt.axes()
Pla.plot(x,y)
plt.show()