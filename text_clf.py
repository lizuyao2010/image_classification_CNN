from dataUtils import *
from sklearn import metrics
import tensorflow as tf
from six.moves import range, reduce


class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, word2idx, label2id):
        super(DataLoader, self).__init__()
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


class LinearClassfier(object):
    """docstring for LinearClassfier"""
    def __init__(self, batch_size,vocab_size, num_classes,session,initializer=tf.random_normal_initializer(stddev=0.1)
        ,optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),name='linearclf'):
        super(LinearClassfier, self).__init__()
        self.batch_size = batch_size
        self.num_classes=num_classes
        self.initializer=initializer
        self.optimizer=optimizer
        self.vocab_size=vocab_size
        self.descriptions = tf.placeholder(tf.float32,[None,self.vocab_size],name="descriptions")
        self.labels = tf.placeholder(tf.int32,[None,self.num_classes],name="labels")
        self.name=name
        with tf.variable_scope(self.name):
            self.word_emb=tf.Variable(self.initializer([self.vocab_size,self.num_classes]),name='word_emb')
        logits=tf.matmul(self.descriptions,self.word_emb)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits,tf.cast(self.labels,tf.float32),name="cross_entropy")
        cross_entropy_sum=tf.reduce_sum(cross_entropy,name="cross_entropy_sum")
        loss_op=cross_entropy_sum
        grads_and_vars=self.optimizer.compute_gradients(loss_op)
        train_op=self.optimizer.apply_gradients(grads_and_vars,name="train_op")
        predict_op = tf.argmax(logits,1,name="predict_op")
        self.loss_op=loss_op
        self.predict_op=predict_op
        self.train_op=train_op
        init_op=tf.initialize_all_variables()
        self.session=session
        self.session.run(init_op)

    def batch_fit(self, descriptions, labels):
        """Runs the training algorithm over the passed batch

        Args:
            descriptions: Tensor (None, vocab_size)
            labels: Tensor (None, num_classes)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self.descriptions: descriptions, self.labels: labels}
        loss, _ = self.session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, descriptions):
        """Predicts answers as one-hot encoding.

        Args:
            descriptions: Tensor (None, vocab_size)
            labels: Tensor (None, num_classes)

        Returns:
            labels: Tensor (None, num_classes)
        """
        feed_dict = {self.descriptions: descriptions}
        return self.session.run(self.predict_op, feed_dict=feed_dict)

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
    tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
    tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
    tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
    tf.flags.DEFINE_integer("random_state", None, "Random state.")
    tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
    tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
    FLAGS = tf.flags.FLAGS
    df=pd.read_csv('records.csv', header=None)
    df[1]=df[1].str.lower()
    word2idx=build_text_vocab(df)
    label2id=build_label_vocab(df)
    dataLoader=DataLoader(word2idx,label2id)
    data,labels=dataLoader.vectorize_text(df)
    print "data shape: ", data.shape
    print "labels shape:",labels.shape
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    batch_size = FLAGS.batch_size
    n_train=len(data)
    train_labels = np.argmax(labels, axis=1)
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]
    with tf.Session() as sess:
        model=LinearClassfier(batch_size,dataLoader.vocab_size,dataLoader.num_classes,sess,optimizer=optimizer)
        for t in range(1,FLAGS.epochs):
            np.random.shuffle(batches)
            total_cost=0.0
            for start,end in batches:
                data_batch=data[start:end]
                label_batch=labels[start:end]
                cost_t=model.batch_fit(data_batch,label_batch)
                total_cost+=cost_t
            if t%FLAGS.evaluation_interval==0:
                train_preds=[]
                for start in range(0,n_train,batch_size):
                    end=start+batch_size
                    data_batch=data[start:end]
                    label_batch=labels[start:end]
                    pred_batch=model.predict(data_batch)
                    train_preds+=list(pred_batch)
                train_acc=metrics.accuracy_score(np.array(train_preds),train_labels)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('-----------------------')

if __name__=='__main__':
    main()



