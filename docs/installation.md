(sec_installation)=
# Installation


Install sc2ts from PyPI:

```sh
python -m pip install sc2ts
```

This installs the minimal version of sc2ts which provides the Python interfaces to
perform
{ref}`sec_arg_analysis`
and
{ref}`sec_alignments_analysis`.

To perform {ref}`sec_inference` we require some additional dependencies which are
installed as follows:

```sh
python -m pip install 'sc2ts[inference]'
```

