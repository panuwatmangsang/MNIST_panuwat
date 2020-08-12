# import matplotlib.pyplot as plt
# from sklearn import datasets
# digit_dataset = datasets.load_digits()

# จำนวนรูปข้อมูล
# print(digit_dataset['images'].shape) 

# ข้อมูลเลขลายมือที่เก็บไว้แบบอ้าง key
# print(digit_dataset['target_names'])
# หรือแบบไม่อ้าง key
# print(digit_dataset.target_names)

# ดูรูปตัวเลขตำแหน่งแรก
# print(digit_dataset.images[0])
# ขนาดตัวเลขที่เก็บ
# print(digit_dataset.images[0].shape)

# แสดงข้อมูล
# print(digit_dataset.target[0])
# แสดงรูปตัวเลขที่กำหนดและกำหนดสี cmap = color map
# plt.imshow(digit_dataset.images[0],cmap=plt.get_cmap('gray'))
# plt.show()

#####################################################################################################
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


mnist_raw = loadmat("mnist-original.mat")
# print(mnist_raw)   

mnist = {
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
# print(mnist["data"].shape)

x,y = mnist["data"],mnist["target"]
# print(x.shape)
# print(y)

# ดึงค่าในตำแหน่งของ x
number=x[10000]
# ภาพที่แสดงผลออกมามีขราด 28x28 px โดยแปลงให้เป็น Array 2D
# number_image=number.reshape(28,28)
# print(y[10000])
# plt.imshow(number_image,cmap=plt.cm.binary,interpolation="nearest")
# plt.show()

#####################################################################
# PCA 
x_train, x_test, y_train, y_test = train_test_split(mnist["data"],mnist["target"], random_state=0)
# print("Before = " ,x_train.shape)
pca=PCA(.95)
# แปลงให้เป็น Array 2D
data=x_train=pca.fit_transform(x_train)
# print("After = " ,x_train.shape)
# แปลงให้เป็นข้อมูลเดิม
result=pca.inverse_transform(data)

# show image , figure = ฉากวาดกราฟ , subplot = กราฟย่อย
plt.figure(figsize=(8,4))
# รูปที่ยังไม่ได้ลดขนาด 784
plt.subplot(1,2,1)
plt.imshow(
    mnist["data"][0].reshape(28,28),
    cmap=plt.cm.gray,
    interpolation="nearest"
)
plt.xlabel("784 Features")
plt.title("Original")
# รูปที่ลดขนาดแล้ว 95% -> 154
plt.subplot(1,2,2)
plt.imshow(
    result[0].reshape(28,28),
    cmap=plt.cm.gray,
    interpolation="nearest"
)
plt.xlabel("154 Features")
plt.title("PCA Image")
plt.show()
