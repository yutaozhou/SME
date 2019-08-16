Installation
=============

How to install SME:

Prerequisites: 
    - libgfortran3
    - gcc

0. (optional) create a virtual environment, and activate it.
    - download the environment.yml from https://github.com/AWehrhahn/SME.git
    - cd to the download location
    - conda env create -f environment.yml
    - source activate pysme

1. Use pip
    - For the "stable" version (this is not regularly updated at the moment)
        - pip install pysme-astro
    - For the latest version use:
        - pip install git+https://github.com/AWehrhahn/SME.git

2. Download data files as part of IDL SME from http://www.stsci.edu/~valenti/sme.html
    - The atmosphere and nlte data files should be downloaded from the server correctly (assuming it is available) if not you can manually copy them into their respective storage locations in ~/.sme/atmospheres and ~/.sme/nlte_grids
        - atmospheres
            - everything from SME/atmospheres
        - nlte_grids
            - \*.grd from SME/NLTE

3. Run SME
    - python minimum.py
