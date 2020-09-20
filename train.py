import glob, os
import random
import os.path
import time
from shutil import copyfile
from subprocess import call
import cv2
import shutil
import sys
from xml.dom import minidom
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
need_file=["content/result","cfg.person"]
#--------------------------------------------------------------------
xmlFolder = "/content/videoXml"
imgFolder = "/content/allVideo"
saveYoloPath = "/content/result"
class_folder = "content/class_state"
classList = { "person":0 }
modelYOLO = "yolov2-tier"  #yolov2-tier or yolov2-tiny
testRatio = 0.2
cfgFolderName = "cfg.person"
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

negative_images = True  #treate images with no xml files as negative images
numBatch = 64
numSubdivision = 1
darknetEcec = "/gdrive/My Drive/darknet/darknet"

#---------------------------------------------------------------------
if(os.path.exists("garbage.txt")):
  f = open("garbage.txt", 'r')
  seq = f.readline()
  garbage=int(seq)
  f.close()
else:
  garbage = 0

try:
  for i in need_file:
      os.rename(i, i+str(garbage))
      shutil.move(i+str(garbage),"garbage")
except:
  imustdo=0

garbage += 1

f = open("garbage.txt", 'w+')
f.writelines(str(garbage))
f.close()



if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)

def downloadPretrained(url):
    import wget
    print("Downloading the pretrained model darknet53.conv.74, please wait.")
    wget.download(url)

def transferYolo( xmlFilepath, imgFilepath, labelGrep=""):
    global imgFolder

    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = basename(img_file)
    yoloFilename = os.path.join(saveYoloPath ,img_filename + ".txt")

    if(xmlFilepath is not None or negative_images is False):
        #print(imgFilepath)
        img = cv2.imread(imgFilepath)
        imgShape = img.shape
        #print (img.shape)
        img_h = imgShape[0]
        img_w = imgShape[1]

        labelXML = minidom.parse(xmlFilepath)
        labelName = []
        labelXmin = []
        labelYmin = []
        labelXmax = []
        labelYmax = []
        totalW = 0
        totalH = 0
        countLabels = 0

        tmpArrays = labelXML.getElementsByTagName("filename")
        for elem in tmpArrays:
            filenameImage = elem.firstChild.data

        tmpArrays = labelXML.getElementsByTagName("name")
        for elem in tmpArrays:
            labelName.append(str(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmin")
        for elem in tmpArrays:
            labelXmin.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("ymin")
        for elem in tmpArrays:
            labelYmin.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmax")
        for elem in tmpArrays:
            labelXmax.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("ymax")
        for elem in tmpArrays:
            labelYmax.append(int(elem.firstChild.data))


    with open(yoloFilename, 'a') as the_file:
        if(xmlFilepath is not None or negative_images is False):
            i = 0
            for className in labelName:
                if(className==labelGrep or labelGrep==""):
                    classID = 0
                    x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w 
                    y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
                    w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
                    h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

                    the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                    i += 1

        else:
            the_file.write('')

    the_file.close()

#---------------------------------------------------------------
fileCount = 0

print("Step 1. Transfer VOC dataset to YOLO dataset.")
if(negative_images is True):
    print("If there is no xml with same names for the images, those images will be treated as negative images.")

for file in os.listdir(imgFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
        imgfile = os.path.join(imgFolder ,file)
        xmlfile = os.path.join(xmlFolder ,filename + ".xml")

        if(os.path.isfile(xmlfile)):
            print("id:{}".format(fileCount))
            print("processing {}".format(imgfile))
            print("processing {}".format(xmlfile))
            fileCount += 1

            transferYolo( xmlfile, imgfile, "")
            copyfile(imgfile, os.path.join(saveYoloPath ,file))

        elif(negative_images is True):
            transferYolo( None, imgfile, "")
            copyfile(imgfile, os.path.join(saveYoloPath ,file))

print("        {} images transered.".format(fileCount))
# step2 ---------------------------------------------------------------
if testRatio!=0:
  Train_times = range(int(1/testRatio))
  Train_times = [i+1 for i in Train_times]
  print('\n',len(Train_times),' fold cross validation')

fileList = []
for file in os.listdir(saveYoloPath):
  filename, file_extension = os.path.splitext(file)
  file_extension = file_extension.lower()

  if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
    fileList.append(os.path.join(saveYoloPath ,file))

buble_list=[]

for i in range(len(fileList)):
  target=int(fileList[i].split('(')[1].split(')')[0])
  for j in range(len(fileList)-i-1):
    buff = int(fileList[i+j+1].split('(')[1].split(')')[0])
    if(target>buff):
      print("--------------------")
      print(fileList[i],"***",fileList[i+j+1])
      temp=str(fileList[i])
      fileList[i]=str(fileList[i+j+1])
      print(fileList[i],"***",temp)
      fileList[i+j+1]=str(temp)
      print(fileList[i],"***",fileList[i+j+1])
  #print(i)    

print(fileList)

fall_state = []
Not_fall_state = []
temp_state = []

for i in os.listdir(imgFolder):
  buff = i.replace('jpg','txt')
  f = open(class_folder+'/'+buff)
  buff = f.readlines()[0]
  if (buff) =="0":
    temp_state.append(i)
  elif (buff) =="1":
    fall_state.append(i)
  else:
    Not_fall_state.append(i)

  f.close()



data_state_round = [fall_state,Not_fall_state,temp_state]
data_len = min(len(fall_state),len(Not_fall_state),len(temp_state))

for i in range(len(data_state_round)):
  data_state_round[i] = data_state_round[i][:data_len]

for i in range(data_len):
  for j in data_state_round:
    j[i] = saveYoloPath+'/'+j[i]
    
for m in Train_times:
  Index=m
  print("-------------------Index:",Index,"-------------------")
  cfgFolder=cfgFolderName
  cfgFolder = str(Index)+'_'+cfgFolder
  outputTrainFile = cfgFolder + "/train.txt"
  outputTestFile = cfgFolder + "/test.txt"

  print("Step 2. Create YOLO cfg folder and split dataset to train and test datasets.")
  if not os.path.exists(cfgFolder):
      os.makedirs(cfgFolder)
  for p in data_state_round:
    
    fileList = p
    testCount = int(len(fileList) * testRatio)

    if (testRatio!=0):
      trainCount = testCount*((1/testRatio)-1)
      vaild_data = int(testCount*int(1/testRatio))
      fileList=fileList[:vaild_data]
    else:
      trainCount = int(len(fileList))

    a = range(len(fileList))   

    test_data = range(int((Index-1)*testCount),int(Index*testCount))
    train_data = [x for x in a if x not in test_data]
    

    with open(outputTrainFile, 'a') as the_file:
        for i in train_data:
          the_file.write(fileList[i] + "\n")

    the_file.close()

    with open(outputTestFile, 'a') as the_file:
        for i in test_data:
            the_file.write(fileList[i] + "\n")
    the_file.close()
    

  print("        Train dataset(x"+str(len(data_state_round))+"):{} images".format(len(train_data)))
  print("        Test dataset(x"+str(len(data_state_round))+"):{} images".format(len(test_data)))
  
  # step3 -------------------------------------------

  print("Step 3. Generate data & names files under "+cfgFolder+ " folder, and update YOLO config file.")

  classes = len(classList)

  if not os.path.exists(os.path.join(cfgFolder ,"weights")):
      os.makedirs(os.path.join(cfgFolder ,"weights"))
      print("all weights will generated in here: " + os.path.join(cfgFolder ,"weights"))

  with open(os.path.join(cfgFolder ,cfg_obj_data), 'w') as the_file:
      the_file.write("classes= " + str(classes) + "\n")
      the_file.write("train  = " + os.path.join(cfgFolder ,"train.txt") + "\n")
      the_file.write("valid  = " + os.path.join(cfgFolder ,"test.txt") + "\n")
      the_file.write("names = " + os.path.join(cfgFolder ,"obj.names") + "\n")
      the_file.write("backup = " + "content/drive/MyDrive/weights_save")
      
  the_file.close()

  with open(os.path.join(cfgFolder ,cfg_obj_names), 'w') as the_file:
      for className in classList:
          the_file.write(className + "\n")

  the_file.close()

  # step4 ----------------------------------------------------

  print("Step 4. Start to train the YOLO model.")



  classNum = len(classList)
  filterNum = (classNum + 5) * 3

  if(modelYOLO == "yolov2-tiny"):
      fileCFG = "yolov2-tiny.cfg"

  else:
      fileCFG = "yolov2-tier.cfg"

  with open(os.path.join("./makeYOLOv3/cfg",fileCFG)) as file:
      file_content = file.read()

  file.close

  file_updated = file_content.replace("{BATCH}", str(numBatch))
  file_updated = file_updated.replace("{SUBDIVISIONS}", str(numSubdivision))
  file_updated = file_updated.replace("{FILTERS}", str(filterNum))
  file_updated = file_updated.replace("{CLASSES}", str(classNum))

  file = open(os.path.join(cfgFolder,fileCFG), "w")
  file.write(file_updated)
  file.close

  executeCmd = darknetEcec + " detector train " + os.path.join(cfgFolder,"obj.data") + " " \
      + os.path.join(cfgFolder,fileCFG) + " darknet53.conv.74"

  print("        please copy and paste to run the darknet training command below:")
  print("          " + executeCmd)
  print("")
  print("        after training, you can find all the weights files here:" + os.path.join(cfgFolder ,"weights"))

  time.sleep(3)


# step5 ----------------------------------------------------
  path = str(Index)+'_'+"test_img"
  if not os.path.isdir(path):
      os.mkdir(path)
  f = open(str(Index)+'_'+'cfg.person/test.txt')
  for i in f:
      location = i.split('\n')[0]
      record = i.split('/')
      record = record[3].split('\n')[0]
      shutil.copyfile(location,path+'/'+record)
  f.close()
