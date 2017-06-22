from setuptools import setup

setup(
    name="windbag",
    version="0.1",
    install_requires=[
        "numpy",
    ],
    extras_require={'tensorflow': ['tensorflow'],
                    'tensorflow with gpu': ['tensorflow-gpu']},
)
