from setuptools import setup, find_packages

setup(
    name="faceAuth",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'dlib',
        'face_recognition',
        'numpy'
    ],
    author="Amos Merber",
    author_email="amosmerber@gmail.com",
    description="A face recognition library for authentication",
    url="https://github.com/merber520/faceAuth",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
