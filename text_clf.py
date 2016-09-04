from dataUtils import *

class TextClf(object):
    """docstring for TextClf"""
    def __init__(self, word2idx, label2id):
        super(TextClf, self).__init__()
        self.word2idx = word2idx
        self.label2id = label2id
        self.vocab_size=len(word2idx)
        self.num_classes=len(label2id)
        print "vocab_size",self.vocab_size
        print "num_classes ", self.num_classes

    def vectorize_text(self,df):
        data=[]
        labels=[]
        for row,label in zip(df[1],df[2]):
            vec=np.zeros(self.vocab_size)
            label_vec=np.zeros(self.num_classes)
            if type(row)==str:
                words=tokenize(row)
                for word in words:
                    vec[self.word2idx[word]]=1
            label_vec[self.label2id[label]]=1
            data.append(vec)
            labels.append(label_vec)
        return np.array(data),np.array(labels)

def build_label_vocab(df):
    labels=sorted(set(df[2]))
    label2id=dict((label,i) for i,label in enumerate(labels))
    return label2id

def build_text_vocab(df):
    text_rows=df[1]
    vocab = sorted(reduce(lambda x, y: x | y, (set(tokenize(row)) for row in text_rows if type(row)==str)))
    word2idx = dict((c, i) for i, c in enumerate(vocab))
    return word2idx



def main():
    df=pd.read_csv('records.csv', header=None)
    df[1]=df[1].str.lower()
    word2idx=build_text_vocab(df)
    label2id=build_label_vocab(df)
    model=TextClf(word2idx,label2id)
    data,labels=model.vectorize_text(df)
    print "data shape: ", data.shape
    print "labels shape:",labels.shape
    # data=vectorize_text(df,word2idx,label2id)


if __name__=='__main__':
    main()



