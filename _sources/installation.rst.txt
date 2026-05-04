Installation
============

.. note::

    Python must already be installed in your current working environment. Instructions to create a new environment for planetsca are available :ref:`here <setting-up-an-environment>`.

.. attention::

    For data access, you must have an `API key <https://developers.planet.com/quickstart/apis/#find-your-api-key>`_ from Planet.

Installing with pip
-------------------

To install the package, use

.. code-block:: bash

    pip install --extra-index-url https://test.pypi.org/simple planetsca

.. _setting-up-an-environment:

Setting up an environment
-------------------------

Setting up a conda environment in the terminal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find detailed instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.

1. Open a terminal
2. Create a new environment called "planetenv" with python version 3.8 or
   greater:

.. code-block:: bash

   conda create -n planetenv python=3.9


3. Activate your new environment:

.. code-block:: bash

   conda activate planetenv


4. Install planetsca from pip

.. code-block:: bash

    pip install --extra-index-url https://test.pypi.org/simple planetsca


To use jupyter notebooks with this conda environment:

1. Activate your new environment:

.. code-block:: bash

    conda activate planetenv


2. Install ipykernel:

.. code-block:: bash

    pip install --user ipykernel


3. Connect this environment to notebooks:

.. code-block:: bash

   python -m ipykernel install --user --name=planetenv


4. When you start a jupyter notebook, you can now select the ``planetenv``
   environment kernel

Setting up a Virtual Environment (VENV) on VSCode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a VENV is recommended for this project as it ensures that there are no
package conflicts and that troubleshooting is much easier. The following
instructions are summarized from
[here](https://code.visualstudio.com/docs/python/environments).

1. Open Command Palette (Ctrl + Shift + P)
2. Select Venv
3. Select Desired Interpreter Path (I use 3.12.2, minimum version is 3.8)
4. Notification should show up on the bottom right corner titled "Creating
   environment (Show logs): Creating venv...
5. The venv is set up and activated if you see " _Python Version_
   (.'venv':venv)"

.. note::

    After setting up a venv once, VScode will automatically start up the virtual environment alongside with VScode. There is no need to repeat these steps unless you do not see step 5.

More information on using environments in VScode is available `here <https://code.visualstudio.com/docs/python/environments>`_

Installing for contributors
---------------------------

We welcome new contributions, improvements, and bug fixes! New features and ideas for the package within the scope of snow remote sensing using Planet imagery are also welcome. Please create an issue to discuss a new feature of bug fix.

To contribute code addressing new features or bug fixes, follow these steps:

1. Clone the repository (or a fork of the repository)

.. code-block:: bash

   git clone https://github.com/DSHydro/planetsca.git


2. Go to the repository

.. code-block:: bash

   cd planetsca


3. Install the development version of planetsca

.. code-block:: bash

   pip install -e ".[dev]"


4. Setup pre-commit

.. code-block:: bash

   pre-commit install


5. Create a feature branch with a descriptive name (e.g. ``bug-fix-planet-api-change``). This command will also switch you to this new branch.

.. code-block:: bash

   git checkout -b new-branch-name


6. After making your changes, run pre-commit before pushing those changes to your new branch (example below)

.. code-block:: bash

   git add your_updated_file_here.py
   pre-commit
   .
   .
   .
   git commit -m 'Bug fixes for planet api change'
   git push -u origin bug-fix-planet-api-change


7. Then submit a pull request (PR) from the feature branch of your fork to ``DSHydro/planetsca:main``

8. The PR will be reviewed by at least one maintainer, discussed, then merged


Additional information
----------------------

This package was developed from `code by Kehan Yang <https://github.com/KehanGit/High_resolution_snow_cover_mapping>`_

Check out the package on TestPyPi: `PyPi <https://test.pypi.org/project/planetsca/>`_

Pre-trained Models hosted on Hugging Face: `Models <https://huggingface.co/geo-smart/planetsca_models>`_

Sample Data hosted on Hugging Face: `Data <https://huggingface.co/datasets/geo-smart/planetsca_datasets>`_

Find your API Key for accessing Planet imagery: `Planet <https://developers.planet.com/quickstart/apis/#find-your-api-key>`_
