from distutils.core import setup

setup(
    name="rubiksolver",
    version="0.0",
    description="An end-to-end Rubik's Cube Solver (from loading input images of the cube to outputting a step-by-step solution).",
    author="Paul Kepley",
    author_email="pakepley@gmail.com",
    url="https://github.com/pkepley/rubiksolver",
    package_dir = {'' : 'src'},
    packages=['rubiksolver']
)
