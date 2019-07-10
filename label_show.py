import h5py
import matplotlib.pylab as plt
import numpy as np
g=h5py.File("TrainData2_100m_110m.hdf5","r+")
#label_set=g.create_dataset(name="/mylabelset2",shape=(5000,1601))
for name in g:
    print(name)
#print(g["mydataset2"][:,:])
#print(g["mylabelset2"][:,:])
HEIGHT = g['mylabelset1'].shape[0]
WEIGHT = g['mylabelset1'].shape[1]
print(HEIGHT,WEIGHT)
#g['mylabelset1'][:,:]=0
user_index = np.linspace(80, 1520, 19,dtype=int)

"""
for j in range(100000):
    max_index = int(np.argmax(g['mylabelset1'][j,user_index])*80 + 80)
    print(max_index)
    g['mylabelset1'][j, :] = 0
    for h in range(51):
        if g['mydataset1'][j,max_index+h] > 6:
            g['mylabelset1'][j, max_index + h] = 1
        else:
            break
    for w in range(51):
        if g['mydataset1'][j,max_index-w] > 6:
            g['mylabelset1'][j, max_index - w] = 1
        else:
            break

g['mylabelset1'][:,86:105]+=2
g['mylabelset1'][:,277:297]+=2
g['mylabelset1'][:,395:408]+=2
g['mylabelset1'][:,611:638]+=2
g['mylabelset1'][:,944:952]+=2
g['mylabelset1'][:,961:990]+=2
g['mylabelset1'][:,1000:1009]+=2
g['mylabelset1'][:,1046:1065]+=2
g['mylabelset1'][:,1157:1180]+=2
g['mylabelset1'][:,1195:1205]+=2
"""

"""
for j in range(100000):
    max_index = int(np.argmax(g['mydataset1'][j,user_index])*80 + 80)
    print(max_index)
    for h in range(51):
        if g['mydataset1'][j,max_index+h] > 6:
            g['mylabelset1'][j, max_index + h] = 1
        else:
            break
    for w in range(51):
        if g['mydataset1'][j,max_index-w] > 6:
            g['mylabelset1'][j, max_index - w] = 1
        else:
            break
"""
"""
for i in range(WEIGHT):
    print(i,g['mydataset1'][100,i])
    if i>=80 and i<1600 and i%80==0:
        print("=========================================",i,g['mydataset1'][100,i])
"""


#g['mylabelset2'][:,:]=0
"""
g['mylabelset3'][:,1193:1226]=2
g['mylabelset3'][:,1379:1407]=2
g['mylabelset3'][:,1592:1600]=2
"""
"""
g['mylabelset2'][:,72:88]=2
g['mylabelset2'][:,236:251]=2
g['mylabelset2'][:,360:376]=2
g['mylabelset2'][:,444:458]=2
g['mylabelset2'][:,706:734]=2
g['mylabelset2'][:,856:874]=2
g['mylabelset2'][:,1052:1062]=2
g['mylabelset2'][:,1174:1206]=2
g['mylabelset2'][:,1300:1323]=2
g['mylabelset2'][:,1524:1547]=2
"""
"""
g['mylabelset1'][:,84:109]+=2
g['mylabelset1'][:,152:176]+=2
g['mylabelset1'][:,279:301]+=2
g['mylabelset1'][:,390:411]+=2
g['mylabelset1'][:,615:640]+=2
g['mylabelset1'][:,944:954]+=2
g['mylabelset1'][:,960:992]+=2
g['mylabelset1'][:,1000:1009]+=2
g['mylabelset1'][:,1045:1066]+=2
g['mylabelset1'][:,1157:1180]+=2
"""
"""
for i in range(100000):
    if g['mylabelset1'][i,83]==1 and g['mylabelset1'][i,109]==1:
        g['mylabelset1'][i, 84:109]=1
    if g['mylabelset1'][i,278]==1 and g['mylabelset1'][i,301]==1:
        g['mylabelset1'][i, 279:301]=1
    if g['mylabelset1'][i,389]==1 and g['mylabelset1'][i,411]==1:
        g['mylabelset1'][i, 390:411]=1
    if g['mylabelset1'][i,614]==1 and g['mylabelset1'][i,640]==1:
        g['mylabelset1'][i, 615:640]=1
    if g['mylabelset1'][i,963]==1 and g['mylabelset1'][i,982]==1:
        g['mylabelset1'][i, 964:982]=1
    if g['mylabelset1'][i,1049]==1 and g['mylabelset1'][i,1057]==1:
        g['mylabelset1'][i, 1050:1057]=1
    if g['mylabelset1'][i,1195]==1 and g['mylabelset1'][i,1205]==1:
        g['mylabelset1'][i, 1196:1205]=1
"""
"""
for i in range(HEIGHT):
    TX_index = user_index[np.argmax(np.subtract(g['mydataset2'][i, user_index],mean_base))]
    g['mylabelset2'][i,TX_index-35:TX_index+36]=1
"""
for i in range(WEIGHT):
    print(g['mydataset1'][0,i])
def show():
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(g['mydataset1'][0:100,:],interpolation='nearest',aspect='auto')
    plt.subplot(2,1,2)
    plt.imshow(g['mylabelset1'][0:100,:],interpolation='nearest',aspect='auto')
    plt.show()
if __name__ == "__main__":
    #plt.imshow(g['mydataset2'][0:500, :], interpolation='nearest', aspect='auto')
    #plt.show()
    show()
    pass