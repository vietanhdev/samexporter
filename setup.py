from setuptools import find_packages, setup


def get_long_description():
    """Read long description from README"""
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name="samexporter",
    version="0.2.0",
    description="Exporting Segment Anything models different formats",
    author="Viet Anh Nguyen",
    author_email="vietanh.dev@gmail.com",
    url="https://github.com/vietanhdev/samexporter",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
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
            "timm",
        ],
        "dev": ["flake8", "isort", "black", "pre-commit"],
    },
)
