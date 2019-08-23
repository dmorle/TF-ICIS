import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from hilbertCurve import hilbertCurve

dataPath = "../../data/sorted/"
dumpPath = "../../data/subsetData/"

#recursively goes though folders and deletes them
def deleteFolder(path):
	if os.path.isdir(path):
		for file in os.listdir(path):
			if (os.path.isfile(path + file)):
				os.remove(path + file)
			else:
				deleteFolder(path + file + '/')
		os.rmdir(path)
	return

# sets up the nessisary hard drive space
def createFolders():
	for i in range(10):
		subfolder = dumpPath + str(i) + "/"
		os.mkdir(subfolder)
	return

def createSubsets_noOverlap(pathIn, pathOut, hC):
	# getting the image in RAM
	img = None
	with open(pathIn, "rb") as f:
		img = pickle.load(f)

	# checking validity
	imgSize = img.shape[0]
	subNum = hC.size + 1
	if imgSize != img.shape[1] or imgSize%subNum != 0:
		raise Exception("Invalid image shape for the given hilbert curve")
	subSize = int(imgSize/subNum)
	
	# formatting the image and curve
	#img = img/255
	curve = hC.getDPoints()
	for i in range(len(curve)):
		curve[i][0] *= imgSize - subSize
		curve[i][1] *= imgSize - subSize
		curve[i][0] = int(curve[i][0])
		curve[i][1] = int(curve[i][1])

	# initializing the subsets
	subsets = np.empty((subNum*subNum, 1, subSize, subSize))
	for i in range(subNum * subNum):
		subsets[i] = img[curve[i][0]:(curve[i][0] + subSize), curve[i][1]:(curve[i][1] + subSize)]
		subsets[i] = np.reshape(subsets[i], (1, subSize, subSize))

	# dumping the subsets
	with open(pathOut, "wb") as f:
		pickle.dump(subsets, f)

	return

def displayImages(rowNum, colNum, imageArr, labelArr):
	fig, axes = plt.subplots(rowNum, colNum, figsize = [12, 9])
	for i in range(rowNum):
		for j in range(colNum):
			axes[i, j].imshow(imageArr[i*colNum + j])
			axes[i, j].set_title("LABEL = " + str(labelArr[i*colNum + j]))
	plt.show()
	return

def createAllSubsets():
	# getting the memory space ready
	for i in range(10):
		deleteFolder(dumpPath + str(i) + "/")
	createFolders()

	# create all subsets
	for i in range(10): # iterating over all labels
		for file in os.listdir(dataPath + str(i)): # iterating over all images in the label
			createSubsets_noOverlap(
				dataPath + str(i) + "/" + file, 
				dumpPath + str(i) + "/" + file,
				hilbertCurve(0)
			)
	return

def main():
	createAllSubsets()
	return

if __name__ == "__main__":
	main()