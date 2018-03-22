import os
from numpy.distutils.core import setup

setup(
    name = "Python-data-analysis",
    version = "1.0",
    description = ("Toolbox for data analysis in Python" ),
    # Author details
    author = "Romain Escudier",
    author_email = "r.escudier@gmail.com",
    # Choose your license
    license = "GPLv3",
    # What does your project relate to?
    keywords = "data analysis / spectrum / filter",
    # The project's main homepage.
    url = "https://github.com/romain-escudier/python-data-analysis",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['rt_anatools'],
)









