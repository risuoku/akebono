from akebono.dataset.model import Dataset


class GlobalDatasetHolder:
    def __init__(self):
        self._value = {}

    def set(self, ds):
        if not isinstance(ds, Dataset):
            raise TypeError('ds must be Dataset.')
        self._value[ds.name] = ds

    def get(self, name):
        return self._value.get(name)


datasetholder = GlobalDatasetHolder()
