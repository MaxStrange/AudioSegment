__doc__ = """Wrapper for pydub.AudioSegment for additional high level methods.

See the github README for documentation."""

from setuptools import setup

setup(
    name="audiosegment",
    version="0.23.0",
    author="Max Strange",
    author_email="maxfieldstrange@gmail.com",
    description="Wrapper for pydub.AudioSegment for additional methods.",
    install_requires=["pydub", "webrtcvad", "numpy"],
    extras_require={
        'Optimizations': ["scipy"],
        'Mel Frequencies': ["librosa"],
    },
    license="MIT",
    keywords="audio sound pydub",
    url="https://github.com/MaxStrange/AudioSegment",
    py_modules=["audiosegment"],
    packages=["algorithms"],
    python_requires="~=3.4",
    long_description=__doc__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ]
)
