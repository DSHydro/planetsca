Installation
============

**This is a short guide on installing the PlanetSCA library.**

.. note::

    Ensure that Python is installed on your system. For data access, you must have an API key from Planet


To install the package, use

.. code-block:: bash

    pip install --extra-index-url https://test.pypi.org/simple planetsca


Virtual Environment Setup
-------------------------

For easiest installation, use a virtual environment.

1. Open Command Palette (Ctrl + Shift + P)
2. Select Venv
3. Select Desired Interpreter Path (I use 3.12.2, minimum version is 3.8)
4. Notification should show up on the bottom right corner titled "Creating
   environment (Show logs): Creating venv...
5. The venv is set up and activated if you see " *Python Version*
   (.'venv':venv)"
6. *Note* After setting up a venv once, VScode will automatically start up the
   virtual environment alongside with VScode. There is no need to repeat these
   steps unless you do not see step 5.

More information available here: `Guide <https://code.visualstudio.com/docs/python/environments>`_

Development
-----------

1. Clone the repository

.. code-block:: bash

   git clone https://github.com/DSHydro/planetsca.git


2. Go to the repository

.. code-block:: bash

   cd planetsca


3. Install the package

.. code-block:: bash

   pip install -e ".[dev]"


4. Setup pre-commit

.. code-block:: bash

   pre-commit install

Additional information
----------------------

This package is originally modeled after Kehan's repository: `Github <https://github.com/KehanGit/High_resolution_snow_cover_mapping>`_

Check out the package on TestPyPi: `PyPi <https://test.pypi.org/project/planetsca/>`_

Pre-trained Models hosted on Huggingfaces: `Huggingface Models <https://huggingface.co/geo-smart/planetsca_models>`_

Sample Data hosted on Huggingfaces: `Huggingface Data <https://huggingface.co/datasets/geo-smart/planetsca_datasets>`_

Find your API Key from Planet: `Planet <https://developers.planet.com/quickstart/apis/#find-your-api-key>`_
