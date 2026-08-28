from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.2",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "pyzmq",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace==0.1.3",
    ],
    packages=find_packages(),
    zip_safe=False,
)
