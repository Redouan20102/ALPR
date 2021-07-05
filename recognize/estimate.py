from matplotlib import pyplot
import pandas as pd


data_file = r'../PFE_final/attributes_modified.csv'

df = pd.read_csv(data_file).dropna()

fields = ["number_of_holes", "horizental_symetry", "vertical_symetry", "mean_vtd",
          "stddev_vtd", "mean_htd", "stddev_htd", "bottom_line_density", "Class"]

colors = ['red', 'blue', 'cyan', 'black', 'yellow', '#ff007f', 'grey', 'orange', 'purple', 'green']
x10 = [df.stddev_vtd[i] for i in range(146)]
x11 = [df.stddev_vtd[i] for i in range(147, 231)]
x12 = [df.stddev_vtd[i] for i in range(232, 350)]
x13 = [df.stddev_vtd[i] for i in range(351, 468)]
x14 = [df.stddev_vtd[i] for i in range(469, 641)]
x15 = [df.stddev_vtd[i] for i in range(642, 761)]
x16 = [df.stddev_vtd[i] for i in range(762, 886)]
x17 = [df.stddev_vtd[i] for i in range(887, 1007)]
x18 = [df.stddev_vtd[i] for i in range(1008, 1128)]
x19 = [df.stddev_vtd[i] for i in range(1129, 1219)]

y10 = [df.stddev_htd[i] for i in range(146)]
y11 = [df.stddev_htd[i] for i in range(147, 231)]
y12 = [df.stddev_htd[i] for i in range(232, 350)]
y13 = [df.stddev_htd[i] for i in range(351, 468)]
y14 = [df.stddev_htd[i] for i in range(469, 641)]
y15 = [df.stddev_htd[i] for i in range(642, 761)]
y16 = [df.stddev_htd[i] for i in range(762, 886)]
y17 = [df.stddev_htd[i] for i in range(887, 1007)]
y18 = [df.stddev_htd[i] for i in range(1008, 1128)]
y19 = [df.stddev_htd[i] for i in range(1129, 1219)]


# plot the histogram
zero = pyplot.scatter(x10, color=colors[0], y=y10)
one = pyplot.scatter(x11, color=colors[1], y=y11)
two = pyplot.scatter(x12, color=colors[2], y=y12)
three = pyplot.scatter(x13, color=colors[3], y=y13)
four = pyplot.scatter(x14, color=colors[4], y=y14)
five = pyplot.scatter(x15, color=colors[5], y=y15)
six = pyplot.scatter(x16, color=colors[6], y=y16)
seven = pyplot.scatter(x17, color=colors[7], y=y17)
eight = pyplot.scatter(x18, color=colors[8], y=y18)
nine = pyplot.scatter(x19, color=colors[9], y=y19)


pyplot.legend((zero, one, two, three, four, five, six, seven, eight, nine),
           ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
pyplot.xlabel("stddev_vtd")
pyplot.ylabel("stddev_htd")
pyplot.savefig('plot.pdf')

pyplot.show()



'''
data_file = r'../dataset/n.csv'

df = pd.read_csv(data_file).dropna()

fields = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "tch1", "tch2", "tch3", "tch4", "tch5", "tch6", "kr", "n",
          "Class"]

x10 = [df.kr[i] for i in range(146)]
x11 = [df.kr[i] for i in range(147, 172)]
x12 = [df.kr[i] for i in range(173, 294)]
x13 = [df.kr[i] for i in range(295, 417)]
x14 = [df.kr[i] for i in range(418, 588)]
x15 = [df.kr[i] for i in range(589, 712)]
x16 = [df.kr[i] for i in range(713, 837)]
x17 = [df.kr[i] for i in range(838, 958)]
x18 = [df.kr[i] for i in range(959, 1077)]
x19 = [df.kr[i] for i in range(1078, 1170)]

y10 = [df.Class[i] for i in range(146)]
y11 = [df.Class[i] for i in range(147, 172)]
y12 = [df.Class[i] for i in range(173, 294)]
y13 = [df.Class[i] for i in range(295, 417)]
y14 = [df.Class[i] for i in range(418, 588)]
y15 = [df.Class[i] for i in range(589, 712)]
y16 = [df.Class[i] for i in range(713, 837)]
y17 = [df.Class[i] for i in range(838, 958)]
y18 = [df.Class[i] for i in range(959, 1077)]
y19 = [df.Class[i] for i in range(1078, 1170)]

# plot the histogram
pyplot.scatter(x10, color='red', y=y10)
pyplot.scatter(x11, color='blue', y=y11)
pyplot.scatter(x12, color='cyan', y=y12)
pyplot.scatter(x13, color='black', y=y13)
pyplot.scatter(x14, color='yellow', y=y14)
pyplot.scatter(x15, color='#ff007f', y=y15)
pyplot.scatter(x16, color='grey', y=y16)
pyplot.scatter(x17, color='orange', y=y17)
pyplot.scatter(x18, color='purple', y=y18)
pyplot.scatter(x19, color='green', y=y19)
pyplot.title('kr')
pyplot.show()
'''