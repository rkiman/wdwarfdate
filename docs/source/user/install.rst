Installation
============

To install *wdwarfdate* you can do it from the github source:

.. code-block:: bash

    git clone https://github.com/rkiman/wdwarfdate.git
    cd wdwarfdate
    python setup.py install


Dependencies
------------

The dependencies of *wdwarfdate* are
`NumPy <http://www.numpy.org/>`_,
`emcee3 <https://emcee.readthedocs.io/en/latest/>`_,

These can be installed using pip:

.. code-block:: bash

    pip install numpy pandas h5py numba "emcee==3.0rc2" tqdm isochrones
