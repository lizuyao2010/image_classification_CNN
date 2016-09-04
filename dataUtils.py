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
from vgg16 import vgg16
import tensorflow as tf

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

def resize_images_in_folder(inputfolder):
    all_image_files=[f for f in listdir(inputfolder) if isfile(join(inputfolder,f)) and f.endswith('.jpg')]
    images=np.zeros((len(all_image_files),224,224,3),dtype=np.uint8)
    for i,f in enumerate(all_image_files):
        print i
        inputpath=join(inputfolder,f)
        index=int(f.rstrip('.jpg'))
        img=resize_image_from_file(inputpath)
        images[index,:,:,:]=img
    return images

def extract_vgg_features_in_folder(inputfolder):
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    images=resize_images_in_folder(inputfolder)
    batch_size=1000
    n=images.shape[0]
    batches = zip(range(0, n-batch_size, batch_size), range(batch_size, n, batch_size))
    batches = [(start, end) for start, end in batches]
    vgg_features=np.zeros((n,4096))
    for start,end in batches:
        images_batch=images[start:end]
        vgg_features_batch=sess.run(vgg.fc2,feed_dict={vgg.imgs: images_batch})
        vgg_features[start:end,:]=vgg_features_batch
    return vgg_features

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
    # print "origin shape",img.shape
    img=resize_image(img)[:,:,:3]
    # print "resized shape",img.shape
    return img

def downloadImages(outputFolder,source):
    df=pd.read_csv(source, header=None)
    for (i,url) in enumerate(df[0]):
        print url
        urllib.urlretrieve(url, outputFolder+str(i)+".jpg")

def main():
    # connect_datapoint('https://test.flaunt.peekabuy.com/api/board/get_jc_product_images_batch/?page=')
    # downloadImages('images/','records.csv')
    # resize_images_in_folder('images','images_resized')
    img_features=extract_vgg_features_in_folder('images')
    print img_features[0]
    img_features.dump("img_features.npy")

if __name__=='__main__':
    main()