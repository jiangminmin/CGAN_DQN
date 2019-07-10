import h5py
import matplotlib.pylab as plt
import numpy as np
import cv2
import os
g=h5py.File("TrainData2_100m_110m.hdf5","r+")

#label_set=g.create_dataset(name="/mylabelset2",shape=(5000,1601))
for name in g:
    print(name)
HEIGHT = g['mydataset1'].shape[0]
WEIGHT = g['mydataset1'].shape[1]

def save_img():
    l=np.array(g['mylabelset1'][0:256,:])
    l = l*255/3
    d=np.array(g['mydataset1'][0:256,:])
    dmin,dmax = d.min(),d.max()
    d = (d-dmin)/(dmax-dmin)*255
    image=np.concatenate((d,l),axis=1)
    cv2.imwrite('test1.jpg',image)
    for i in range(390):
        l = np.array(g['mylabelset1'][256*i:256*(i+1), :])
        l = l * 255 / 3
        d = np.array(g['mydataset1'][256*i:256*(i+1), :])
        dmin, dmax = d.min(), d.max()
        d = (d - dmin) / (dmax - dmin) * 255
        image = np.concatenate((d, l), axis=1)
        cv2.imwrite('/home/jmm/Downloads/5_Deep_Q_Network/train_img_256x1601/train'+str(390+i)+'.jpg', image)

def show():
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(g['mydataset1'][0:100,:],interpolation='nearest',aspect='auto')
    plt.subplot(2,1,2)
    plt.imshow(g['mylabelset1'][0:100,:],interpolation='nearest',aspect='auto')
    plt.show()
def cv_show():
    a = np.array(g['mylabelset1'][0:100, :] * 80, dtype=np.uint8)
    cv2.imshow('haha', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def create_sweep():
    sweep = np.zeros(shape=(100,1601))

    sweep[0:5, 20:141] = 1
    sweep[5:10, 100:221] = 1
    sweep[10:15, 180:301] = 1
    sweep[15:20, 260:381] = 1
    sweep[20:25, 340:461] = 1
    sweep[25:30, 420:541] = 1
    sweep[30:35, 500:621] = 1
    sweep[35:40, 580:701] = 1
    sweep[40:45, 660:781] = 1
    sweep[45:50, 740:861] = 1
    sweep[50:55, 820:941] = 1
    sweep[55:60, 900:1021] = 1
    sweep[60:65, 980:1101] = 1
    sweep[65:70, 1060:1181] = 1
    sweep[70:75, 1140:1261] = 1
    sweep[75:80, 1220:1341] = 1
    sweep[80:85, 1300:1421] = 1
    sweep[85:90, 1380:1501] = 1
    sweep[90:95, 1460:1581] = 1
    sweep[95:100, 20:141] = 1

    sweep[:, 50:70] += 2
    sweep[:, 100:130] += 2
    sweep[:, 200:240] += 2
    sweep[:, 300:350] += 2
    sweep[:, 400:410] += 2
    sweep[:, 500:505] += 2
    sweep[:, 900:920] += 2
    sweep[:, 1000:1019] += 2
    sweep[:, 1145:1166] += 2
    sweep[:, 1357:1380] += 2

    l=np.array(sweep[0:100,:])
    l = l * 255 / 3

    d=np.zeros(shape=(100,1601))
    d=np.array(d[0:100,:])
    image = np.concatenate((d, l), axis=1)

    cv2.imwrite('/media/jiangminmin/00030E7C0003A35C/minmin/pix2pix-tensorflow-master/tools/val/sweep_1.jpg', image)
    plt.imshow(sweep[:,:])
    plt.show()
def read_img():
    img = cv2.imread('/media/jiangminmin/00030E7C0003A35C/minmin/pix2pix-tensorflow-master/facades_test/images/sweep_1-outputs.png',flags=-1)
    img = cv2.resize(img,(1601,100),interpolation=cv2.INTER_CUBIC)
    print(img,img.shape)
    plt.imshow(img,interpolation='nearest',aspect='auto')
    plt.show()
    pass

if __name__ == "__main__":
    #create_sweep()
    save_img()
    pass