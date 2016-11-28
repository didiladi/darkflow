from utils import loader
import numpy as np

class layer(object):
    def __init__(self, num, ltype, *args):
        self.signature = [ltype] + list(args)
        self.number = num
        self.type = ltype

        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.setup(*args) # set attr up
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        if self.type not in src_loader.VAR_LAYER: return
        src_type = type(src_loader)
        if src_type is loader.weights_loader:
            self.load_weights(src_loader)
        else: self.load_ckpt(src_loader)

    def load_weights(self, src_loader):
        val = src_loader([self])
        sig = [self.number, self.signature]
        if val is not None: self.w = val.w
        self.verbalise(val is None, sig)

    def load_ckpt(self, src_loader):
        for var in self.wshape:
            name = str(self.number)
            name += '-' + self.type
            name += '-' + var
            shape = self.wshape[var]
            key = [name, shape]
            val = src_loader(key)
            self.w[var] = val
            self.verbalise(val is None, key)

    def verbalise(self, initialize, sig):
        msg = 'Re-collect'
        if initialize: msg = 'Initialize'
        template = '{:<10} layer {:>3}: {}' 
        print template.format(msg, *sig)


    # For comparing two layers
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

    # Derivative methods
    def setup(self, *args): pass
    def finalize(self): pass 

class avgpool_layer(layer):
    pass

class maxpool_layer(layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class softmax_layer(layer):
    def setup(self, groups):
        self.groups = groups

class dropout_layer(layer):
    def setup(self, p):
        self.h['pdrop'] = {
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': []
        }

class convolutional_layer(layer):
    def setup(self, ksize, c, n, stride, pad, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.stride = stride
        self.pad = pad
        self.dnshape = [n, c, ksize, ksize] # darknet shape
        self.wshape = {
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        }
        if self.batch_norm:
            self.wshape.update({
                'var'  : [n], 
                'scale': [n], 
                'mean' : [n]
            })

    def finalize(self):
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel

class connected_layer(layer):
    def setup(self, input_size, output_size):
        self.wshape = {
            'biases': [output_size],
            'weights': [input_size, output_size]
        }

    def finalize(self):
        weights = self.w['weights']
        if weights is None: return
        weights = weights.reshape(
            self.wshape['weights'])
        self.w['weights'] = weights

"""
Darkop Factory
"""

darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer
}

def create_darkop(num, ltype, *args):
    op_class = darkops.get(ltype, layer)
    return op_class(num, ltype, *args)