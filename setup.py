from distutils.core import setup

setup(
    name="rubiksolver",
    version="0.0",
    description="An end to end solver Rubik's Cube Solver (from input images to solution).",
    author="Paul Kepley",
    author_email="pakepley@gmail.com",
    url="https://github.com/pkepley/rubiksolver",
    package_dir = {'' : 'src'},
    packages=['rubiksolver']
)
