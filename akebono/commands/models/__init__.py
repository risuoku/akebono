from . import train
from . import predict
from . import inspect


# attach models tree

tree = {
    'train': train.Train(),
    'predict': predict.Predict(),
    'inspect': inspect.Inspect(),
}
