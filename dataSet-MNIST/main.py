import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataPath = "C:\\Users\\dmorl\\Desktop\\File_Folder\\coding\\Python\\Computer Vision\\Tensorflow\\TF-ICIS\\dataSet-MNIST\\"

def bytes_from_file(filename, chunksize=4):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break

#recursively goes though folders and deletes them
def deleteFolder(path):
	if os.path.isdir(path):
		for file in os.listdir(path):
			if (os.path.isfile(path + file)):
				os.remove(path + file)
			else:
				deleteFolder(path + file + '\\')
		os.rmdir(path)
	return

# sets up the nessisary hard drive space
def createFolders():
	folder = dataPath + "sorted\\"
	deleteFolder(folder)
	os.mkdir(folder)
	for i in range(10):
		subfolder = folder + str(i) + "\\"
		os.mkdir(subfolder)
	return

def readImages(imagePath, labelPath, imageArr, labelArr):
	# initializing the image array
	headerBytes = 16
	counter = imageNum = height = width = 0
	for byte in bytes_from_file(imagePath, 1):
		if headerBytes > 0:
			headerBytes -= 1
			continue

		imageArr[imageNum][height][width] = int(byte)

		counter += 1
		width = int(counter)%28
		height = int(counter/28)%28
		imageNum = int(counter/28/28)

	# initializing the label array
	headerBytes = 8
	imageNum = 0
	for byte in bytes_from_file(labelPath, 1):
		if headerBytes > 0:
			headerBytes -= 1
			continue

		labelArr[imageNum] = int(byte)

		imageNum += 1
	return

def displayImages(rowNum, colNum, imageArr, labelArr):
	fig, axes = plt.subplots(rowNum, colNum, figsize = [12, 9])
	for i in range(rowNum):
		for j in range(colNum):
			axes[i, j].imshow(imageArr[i*colNum + j])
			axes[i, j].set_title("LABEL = " + str(labelArr[i*colNum + j]))
	plt.show()
	return

def storeData(imageArr, labelArr):
	folder = dataPath + "sorted\\"
	imageCount = [0]*10
	for i in range(60000):
		with open(folder + str(labelArr[i]) + "\\Image_" + "{:04d}".format(imageCount[labelArr[i]]), "wb") as f:
			pickle.dump(imageArr[i], f)

		imageCount[labelArr[i]] += 1
	return

def main():
	imageArr = np.zeros((60000, 28, 28), dtype = np.uint8)
	labelArr = np.zeros((60000), dtype = np.uint8)

	createFolders()
	readImages(dataPath + "train-images.idx3-ubyte", dataPath + "train-labels.idx1-ubyte", imageArr, labelArr)
	storeData(imageArr, labelArr)

	return

if __name__ == "__main__":
	main()