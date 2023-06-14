import argparse
import cv2
import csv
import fnmatch
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--ptns")
args = vars(ap.parse_args())

ptns = int(args["ptns"])

arr = {
	"DATACODE": [],
	"RCALTX": [],
	"RCALTY": [],
	"RCARBX": [],
	"RCARBY": [],
	"LADLTX": [],
	"LADLTY": [],
	"LADRBX": [],
	"LADRBY": [],
	"CXLTX": [],
	"CXLTY": [],
	"CXRBX": [],
	"CXRBY": [],
	"LMLTX": [],
	"LMLTY": [],
	"LMRBX": [],
	"LMRBY": []
	}

for ptn in range(6, ptns + 6):
	num_files = len(fnmatch.filter(os.listdir("GCT{}_IMG".format(str(ptn))), '*.png'))
	for files in range(1, num_files + 1):
		image = cv2.imread("GCT{}_IMG/gct{}_{}.png".format(str(ptn), str(ptn), str(files)))
		
		csvfile = open('GCT{}.csv'.format(ptn))
		
		csvreader = csv.reader(csvfile)
		
		header = []
		header = next(csvreader)
		
		rows = []
		for row in csvreader:
			rows.append(row)
			
		csvfile.close()
		
		ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
		ss.setBaseImage(image)
		ss.switchToSelectiveSearchQuality
		
		rects = ss.process()
		
		arr["DATACODE"].append("gct{}_{}_0.png".format(str(ptn), str(files)))
		
		arr["RCALTX"].append(rows[ptn - 6][1])
		arr["RCALTY"].append(rows[ptn - 6][2])
		arr["RCARBX"].append(rows[ptn - 6][3])
		arr["RCARBY"].append(rows[ptn - 6][4])
		
		arr["LADLTX"].append(rows[ptn - 6][5])
		arr["LADLTY"].append(rows[ptn - 6][6])
		arr["LADRBX"].append(rows[ptn - 6][7])
		arr["LADRBY"].append(rows[ptn - 6][8])
		
		arr["CXLTX"].append(rows[ptn - 6][9])
		arr["CXLTY"].append(rows[ptn - 6][10])
		arr["CXRBX"].append(rows[ptn - 6][11])
		arr["CXRBY"].append(rows[ptn - 6][12])
		
		arr["LMLTX"].append(rows[ptn - 6][13])
		arr["LMLTY"].append(rows[ptn - 6][14])
		arr["LMRBX"].append(rows[ptn - 6][15])
		arr["LMRBY"].append(rows[ptn - 6][16])
		
		for i in range(1, len(rects) + 1):
			output = image.copy()
			
			(x, y, w, h) = rects[i]
			
			region = output[y:y + h, x:x + w]
			
			imagecode = "ssDataset/gct{}_{}_{}.png".format(str(ptn), str(files), str(i))
			arr["DATACODE"].append(imagecode)
			
			RCALTX = rows[ptn - 6][1]
			RCALTY = rows[ptn - 6][2]
			RCARBX = rows[ptn - 6][3]
			RCARBY = rows[ptn - 6][4]
			
			LADLTX = rows[ptn - 6][5]
			LADLTY = rows[ptn - 6][6]
			LADRBX = rows[ptn - 6][7]
			LADRBY = rows[ptn - 6][8]
			
			CXLTX = rows[ptn - 6][9]
			CXLTY = rows[ptn - 6][10]
			CXRBX = rows[ptn - 6][11]
			CXRBY = rows[ptn - 6][12]
			
			LMLTX = rows[ptn - 6][13]
			LMLTY = rows[ptn - 6][14]
			LMRBX = rows[ptn - 6][15]
			LMRBY = rows[ptn - 6][16]
			
			if (RCALTX > x) and (RCALTY > y) and (RCARBX < (x + w)) and (RCARBY < (y + h)):
				newLTX = RCALTX - x
				newLTY = RCALTY - y
				newRBX = RCARBX - x
				newRBY = RCARBY - y
				arr["RCALTX"].append(newLTX)
				arr["RCALTY"].append(newLTY)
				arr["RCARBX"].append(newRBX)
				arr["RCARBY"].append(newRBY)
			else:
				arr["RCALTX"].append(0)
				arr["RCALTY"].append(0)
				arr["RCARBX"].append(0)
				arr["RCARBY"].append(0)
			
			if (LADLTX > x) and (LADLTY > y) and (LADRBX < (x + w)) and (LADRBY < (y + h)):
				newLTX = LADLTX - x
				newLTY = LADLTY - y
				newRBX = LADRBX - x
				newRBY = LADRBY - y
				arr["LADLTX"].append(newLTX)
				arr["LADLTY"].append(newLTY)
				arr["LADRBX"].append(newRBX)
				arr["LADRBY"].append(newRBY)
			else:
				arr["LADLTX"].append(0)
				arr["LADLTY"].append(0)
				arr["LADRBX"].append(0)
				arr["LADRBY"].append(0)

			if (CXLTX > x) and (CXLTY > y) and (CXRBX < (x + w)) and (CXRBY < (y + h)):
				newLTX = CXLTX - x
				newLTY = CXLTY - y
				newRBX = CXRBX - x
				newRBY = CXRBY - y
				arr["CXLTX"].append(newLTX)
				arr["CXLTY"].append(newLTY)
				arr["CXRBX"].append(newRBX)
				arr["CXRBY"].append(newRBY)
			else:
				arr["CXLTX"].append(0)
				arr["CXLTY"].append(0)
				arr["CXRBX"].append(0)
				arr["CXRBY"].append(0)

			if (LMLTX > x) and (LMLTY > y) and (LMRBX < (x + w)) and (LMRBY < (y + h)):
				newLTX = LMLTX - x
				newLTY = LMLTY - y
				newRBX = LMRBX - x
				newRBY = LMRBY - y
				arr["LMLTX"].append(newLTX)
				arr["LMLTY"].append(newLTY)
				arr["LMRBX"].append(newRBX)
				arr["LMRBY"].append(newRBY)
			else:
				arr["LMLTX"].append(0)
				arr["LMLTY"].append(0)
				arr["LMRBX"].append(0)
				arr["LMRBY"].append(0)
				
			cv2.imwrite(imagecode, region)

# Writing to CSV file

f = open("data.csv", 'w')

writer = csv.writer(f)

datarows = len(arr["DATACODE"])

header = ["DATACODE", "RCALTX", "RCALTY", "RCARBX", "RCARBY", "LADLTX", "LADLTY", "LADRBX", "LADRBY", "CXLTX", "CXLTY", "CXRBX", "CXRBY", "LMLTX", "LMLTY", "LMRBX", "LMRBY"]

writer.writerow(header)

for row in range(datarows):
	datacode = arr["DATACODE"][row]
	rcaltx = arr["RCALTX"][row]
	rcalty = arr["RCALTY"][row]
	rcarbx = arr["RCARBX"][row]
	rcarby = arr["RCARBY"][row]
	ladltx = arr["LADLTX"][row]
	ladlty = arr["LADLTY"][row]
	ladrbx = arr["LADRBX"][row]
	ladrby = arr["LADRBY"][row]
	cxltx = arr["CXLTX"][row]
	cxlty = arr["CXLTY"][row]
	cxrbx = arr["CXRBX"][row]
	cxrby = arr["CXRBY"][row]
	lmltx = arr["LMLTX"][row]
	lmlty = arr["LMLTY"][row]
	lmrbx = arr["LMRBX"][row]
	lmrby = arr["LMRBY"][row]
	
	datarow = [datacode, rcaltx, rcalty, rcarbx, rcarby, ladltx, ladlty, ladrbx, ladrby, cxltx, cxlty, cxrbx, cxrby, lmltx, lmlty, lmrbx, lmrby]
	
	writer.writerow(datarow)

f.close()
