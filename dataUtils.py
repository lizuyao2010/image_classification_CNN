import urllib2
import urllib
import json
import csv
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import re

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    return result

def dump_page(records,wr):
    for record in records:
        record=[unicode(item).encode("utf-8") for item in record]
        wr.writerow(record)

def connect_datapoint(url):
    out=open('records.csv','wb')
    wr=csv.writer(out,quoting=csv.QUOTE_ALL)
    for pageNum in range(0,1000):
        req=urllib2.Request(url+str(pageNum))
        response = urllib2.urlopen(req)
        the_page = response.read()
        parsed_page=json.loads(the_page)
        if parsed_page['images']:
            print pageNum
            dump_page(parsed_page['images'],wr)
        else:
            break

def resize_images_in_folder(inputfolder,outputfolder):
    all_image_files=[f for f in listdir(inputfolder) if isfile(join(inputfolder,f)) and f.endswith('.jpg')]
    for f in all_image_files:
        inputpath=join(inputfolder,f)
        img=resize_image_from_file(inputpath)
        outpath=join(outputfolder,f)
        misc.imsave(outpath,img)

def resize_image(img):
    image_h, image_w, _ = np.shape(img)
    shorter_side = min(image_h, image_w)
    scale = 224. / shorter_side
    image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
    img = misc.imresize(img, (image_h, image_w))
    crop_x = (image_w - 224) / 2
    crop_y = (image_h - 224) / 2
    img = img[crop_y:crop_y+224,crop_x:crop_x+224,:]
    return img

def resize_image_from_file(file):
    img=misc.imread(file)
    print "origin shape",img.shape
    img=resize_image(img)[:,:,3]
    # plt.imshow(img)
    # plt.show()
    print "resized shape",img.shape
    return img

def downloadImages(outputFolder,source):
    df=pd.read_csv(source, header=None)
    for (i,url) in enumerate(df[0]):
        print url
        urllib.urlretrieve(url, outputFolder+str(i)+".jpg")

def main():
    # connect_datapoint('https://test.flaunt.peekabuy.com/api/board/get_jc_product_images_batch/?page=')
    downloadImages('images/','records.csv')
    # resize_images_in_folder('images','images_resized')

if __name__=='__main__':
    main()