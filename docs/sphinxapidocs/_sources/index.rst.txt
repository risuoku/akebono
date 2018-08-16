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

.. autofunction:: akebono.preprocessor.get_preprocessor

.. autoclass:: akebono.preprocessor.StatelessPreprocessor
    :inherited-members:

.. autoclass:: akebono.preprocessor.StatefulPreprocessor
    :inherited-members:

.. autoclass:: akebono.preprocessor.Identify
    :inherited-members:

.. autoclass:: akebono.preprocessor.SelectColumns
    :inherited-members:

.. autoclass:: akebono.preprocessor.ExcludeColumns
    :inherited-members:

.. autoclass:: akebono.preprocessor.ApplyStandardScaler
    :inherited-members:

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
