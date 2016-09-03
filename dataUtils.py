import urllib2
import json
import csv
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

out=open('records.csv','wb')
wr=csv.writer(out,quoting=csv.QUOTE_ALL)

def dump_page(records):
    for record in records:
        record=[unicode(item).encode("utf-8") for item in record]
        wr.writerow(record)

def connect_datapoint(url):
    for pageNum in range(0,1000):
        req=urllib2.Request(url+str(pageNum))
        response = urllib2.urlopen(req)
        the_page = response.read()
        parsed_page=json.loads(the_page)
        if parsed_page['images']:
            print pageNum
            dump_page(parsed_page['images'])
        else:
            break

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

def main():
    # connect_datapoint('https://test.flaunt.peekabuy.com/api/board/get_jc_product_images_batch/?page=')
    img=misc.imread('f860ad81767029d91fdc5f18a3d06f8b.jpg')
    print img.shape
    img=resize_image(img)[:,:,:3]
    plt.imshow(img)
    plt.show()
    print img.shape

if __name__=='__main__':
    main()