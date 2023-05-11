from setuptools import find_packages, setup

# Use README for long description
long_description = open("README.md").read()
long_description.replace(
    "(sample_outputs/",
    "(https://raw.githubusercontent.com/vietanhdev/samexporter/main/sample_outputs/",
)

setup(
    name="samexporter",
    version="0.1.0",
    description="Exporting Segment Anything models different formats",
    author="Viet Anh Nguyen",
    author_email="vietanh.dev@gmail.com",
    url="https://github.com/vietanhdev/samexporter",
    long_description=long_description,
    install_requires=[],
    packages=find_packages(),
    extras_require={
        "all": [
            "segment_anything",
            "torch",
            "torchvision",
            "opencv-python",
            "onnx",
            "onnxruntime",
        ],
        "dev": ["flake8", "isort", "black", "pre-commit"],
    },
)
