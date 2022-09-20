import requests, gzip, os, hashlib
import numpy as np
from src.utils import one_hot_encoding

class Mnist:
    def __init__(self):
        #影象檔案的前16個位元組是頭,包含了4個位元組的幻數,4個位元組表示影象數量,4個位元組表示單個影象的行數,4個位元組表示單個影象的列數.
        self.X = self._fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:]
        #標記檔案的前8個位元組是頭,包含了4個位元組的幻數,4個位元組表示標記數量.
        self.Y = self._fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]

    def _fetch(self,url):
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                dat = f.read()
        else:
            with open(fp, "wb") as f:
                dat = requests.get(url).content
                f.write(dat)
        return np.frombuffer(gzip.decompress(dat), dtype = np.uint8).copy()

    def normalization(self):
        self.X = self.X / 255
        return self

    def reshape(self):
        self.X = self.X.reshape((60000,784))
        return self
    
    def one_hot_encode(self):
        self.Y = one_hot_encoding(self.Y,10)
        return self

    def get(self):
        return self.X , self.Y