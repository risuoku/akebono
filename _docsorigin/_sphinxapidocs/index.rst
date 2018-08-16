.. akebono documentation master file, created by
   sphinx-quickstart on Tue Aug 14 16:08:50 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

akebono API documentation for developers
===========================================

Dataset
-----------

.. autofunction:: akebono.dataset.get_dataset

.. autoclass:: akebono.dataset.Dataset
    :inherited-members:

Preprocessor
-------------------

.. autofunction:: akebono.preprocessors.identify

.. autofunction:: akebono.preprocessors.select_columns

.. autofunction:: akebono.preprocessors.exclude_columns

Model
-----------

.. autofunction:: akebono.model.get_model

.. autoclass:: akebono.model.WrappedModel
    :inherited-members:

Operator
-----------

.. autofunction:: akebono.operator.train

.. autofunction:: akebono.operator.predict

Settings
-----------

.. autofunction:: akebono.settings.apply
