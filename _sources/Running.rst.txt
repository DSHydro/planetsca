Running the code
================

.. image:: doc_images/simplify_and_searchfilters_flowchart.png
    :alt: Flow chart of simplify_AOI and Search_Filters modules
    :width: 800px
    :height: 250px
    :scale: 100 %
    :align: center

.. note::

    Use the search_filter module to create a search filter readable by Planet for a data request. Run simplify_AOI to ensure that the geojson will be accepted by Planet.

.. image:: doc_images/data_gathering_flowchart.png
    :alt: Flow chart of data_gathering module
    :width: 800px
    :height: 250px
    :scale: 100 %
    :align: center

.. note::

    Use the data gathering module to create a request to Planet for data. The module will download the data and save it to a specified directory. There is also the option to download sample data.

.. image:: doc_images/data_preparation_flowchart.png
    :alt: Flow chart of simplify_AOI and Search_Filters modules
    :width: 800px
    :height: 250px
    :scale: 100 %
    :align: center

.. note::

    Use the data preparation module to prepare the data for training. The module will create a training dataset and a validation dataset. There is also the option to prepare on already set up csv data.

.. image:: doc_images/model_training_flowchart.png
    :alt: Flow chart of simplify_AOI and Search_Filters modules
    :width: 800px
    :height: 250px
    :scale: 100 %
    :align: center

.. note::

    Use the model training module to train a model on the training dataset. The module will save the model to a specified directory. There is also the option to retrieve a pre-made model.

.. image:: doc_images/data_evaluation_flowchart.png
    :alt: Flow chart of simplify_AOI and Search_Filters modules
    :width: 800px
    :height: 250px
    :scale: 100 %
    :align: center

.. note::

    Use the data evaluation module to evaluate the performance of a model. There are several different prediction functions available.
