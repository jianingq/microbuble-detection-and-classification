import matplotlib.pyplot as plt
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import copy
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

onlyfiles_neg = [f for f in listdir("./ai_cropped/neg/") if f.endswith("npy")]
onlyfiles = [f for f in listdir("./ai_cropped/") if f.endswith("npy")]
print(onlyfiles_neg)
print(onlyfiles)
X = []#np.zeros((14,2))
Y = []
index = 0
X_all = []#np.zeros((len(onlyfiles_neg)+len(onlyfiles)-6,2))
Y_all = []

X_neg_pos = []
X_med_str = []
Y_neg_pos = []
Y_med_str = []

index_list = [24]

for f in onlyfiles_neg:
	#print(index)
	print(f)
	Y.append(0)

	x = np.load("./ai_cropped/neg/"+f)
	#print(np.sum(x))
	#X[index,:len(x)] = np.array(x)
	hist = plt.hist(x, bins=[2,8,20], density=False, facecolor='g')#[2,8,45]#+[i for i in np.arange(10,30,0.05)][5,10,15,20,25,30,35,40,45]

	#X[index,:] = hist[0]
	X.append(hist[0])

	X_all.append(hist[0])
	X_neg_pos.append(hist[0])
	Y_all.append(0)
	Y_neg_pos.append(0)


	index_list.append(index)
	index += 1
	plt.cla()


blurry_list = [24]
for f in onlyfiles:
	x = np.load("./ai_cropped/"+f)

	#X[index,:len(x)] = np.array(x)
	hist = plt.hist(x, bins=[2,8,20], density=False, facecolor='g')#+[i for i in np.arange(10,30,0.05)][5,10,15,20,25,30,35,40,45]


	if (index in blurry_list):
		print(index)
		index_list.append(index)
		X_all.append(hist[0])
		Y_all.append(1)
	else:
		index_list.append(index)
		X_all.append(hist[0])
		Y_neg_pos.append(1)
		X_neg_pos.append(hist[0])
		if(index in [51,50,46,44,42,40,37,30,29,28]):#[24,28,29,30,32,33,35,36,37,39,40,42,44,46,47,50,51]):#
			Y.append(1)
			Y_all.append(1)
			X.append(hist[0])
			Y_med_str.append(0)
			X_med_str.append(hist[0])
		elif(index in [39, 36, 35, 33,32,24,49,48,43,31,26]):#[23,34,38,25,41,27,31,48]):
			Y.append(2)
			Y_all.append(2)
			X.append(hist[0])
			Y_med_str.append(1)
			X_med_str.append(hist[0])
		else:
			Y_all.append(2)
			Y_med_str.append(1)
			X_med_str.append(hist[0])

	#print(index)
	##print(f)
	#print(x)
	#print(np.sum(x))

	index += 1
	plt.cla()


X = np.array(X)
Y = np.array(Y)
X_all = np.array(X_all)
Y_all = np.array(Y_all)

X_neg_pos = np.array(X_neg_pos)
Y_neg_pos = np.array(Y_neg_pos)
X_med_str = np.array(X_med_str)
Y_med_str = np.array(Y_med_str)
def normalize(aa):

	return (aa-np.min(aa))/np.max(aa)

min_0 = np.min(X_all[:,0])
min_1 = np.min(X_all[:,1])
max_0 = np.max(X_all[:,0])
max_1 = np.max(X_all[:,1])
X[:,0] = 100*(X[:,0]-min_0)/max_0
X[:,1] = 100*(X[:,1]-min_1)/max_1

X_med_str[:,0] = 100*(X_med_str[:,0]-min_0)/max_0
X_med_str[:,1] = 100*(X_med_str[:,1]-min_1)/max_1
X_neg_pos[:,0] = 100*(X_neg_pos[:,0]-min_0)/max_0
X_neg_pos[:,1] = 100*(X_neg_pos[:,1]-min_1)/max_1
X_all[:,0] = 100*(X_all[:,0]-min_0)/max_0
X_all[:,1] = 100*(X_all[:,1]-min_1)/max_1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


C = 1.0  # SVM regularization parameter
h = 0.05

# create a mesh to plot in
x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


clf1 = svm.SVC(kernel='linear', C=C).fit(X_med_str, Y_med_str)#svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)#svm.SVC(kernel='linear', C=C).fit(X, y)#svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X, y)#svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)# svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)#svm.LinearSVC(C=C).fit(X, y)#DecisionTreeClassifier().fit(X, y)svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)#
clf2 = svm.SVC(kernel='linear', C=C).fit(X_neg_pos, Y_neg_pos)

#clf1 = svm.SVC(kernel='poly', degree=2, C=C).fit(X_med_str, Y_med_str)#svm.SVC(kernel='linear', C=C).fit(X, y)#svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X, y)#svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)# svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)#svm.LinearSVC(C=C).fit(X, y)#DecisionTreeClassifier().fit(X, y)svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)#
#clf2 = svm.SVC(kernel='poly', degree=2, C=C).fit(X_neg_pos, Y_neg_pos) 

print("done1")

Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z1.reshape(xx.shape)
Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)
print("done 2")

Z1[Z1==1] = 2
Z1[Z1 == 0] = 1
Z1[Z2 == 0] = 0


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
fig, ax = plt.subplots()
# title for the plots
titles = ['LinearSVM']

xx = xx*max_0/100
yy = yy*max_1/100
ax.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
#plt.scatter(X_all[:, 0], X_all[:, 1], c=Y_all, cmap=plt.cm.coolwarm,edgecolors='b')


X_all[:,0] = X_all[:,0] * max_0/100
X_all[:,1] = X_all[:,1] * max_1/100
neg = ax.scatter(X_all[Y_all==0,0], X_all[Y_all==0,1], c=np.array(['b']*sum(Y_all==0)), s= 20)
medium = ax.scatter(X_all[Y_all==1,0], X_all[Y_all==1,1], c=np.array(['c']*sum(Y_all==1)), s= 20)
pos = ax.scatter(X_all[Y_all==2,0], X_all[Y_all==2,1], c=np.array(['m']*sum(Y_all==2)), s= 20)



#for i in range(len(X_all)):#
#	ax.text(X_all[i,0], X_all[i,1], str(index_list[i]),fontsize=8)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# inset axes....
axins = ax.inset_axes([0.7, 0, 0.3, 0.3])
axins.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.8)
axins.scatter(X_all[Y_all==0,0], X_all[Y_all==0,1], c=np.array(['b']*sum(Y_all==0)), s= 20)
axins.scatter(X_all[Y_all==1,0], X_all[Y_all==1,1], c=np.array(['c']*sum(Y_all==1)), s= 20)
axins.scatter(X_all[Y_all==2,0], X_all[Y_all==2,1], c=np.array(['m']*sum(Y_all==2)), s= 20)
# sub region of the original image
x1, x2, y1, y2 = xx.min(), 40, yy.min(), 50
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)

ax.xaxis.label.set_size(20)

## draw a bbox of the region of the inset axes in the parent axes and
## connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.xlabel('number of small bubbles',fontsize=13)
plt.ylabel('number of large bubbles',fontsize=13)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.xticks(())
#plt.yticks(())
#plt.title(titles[0])


plt.legend((pos,medium,neg),
           ('strong +','medium +','negative -'),
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=13)
plt.savefig("cluster.pdf")

plt.show()

