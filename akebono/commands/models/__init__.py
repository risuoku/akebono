from . import train
from . import predict
from . import inspect
from . import init


# attach models tree

tree = {
    'init': init.Init(),
    'train': train.Train(),
    'predict': predict.Predict(),
    'inspect': inspect.Inspect(),
}
