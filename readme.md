



# QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Overview

QCFPDataSet is an open-access dataset that combines classical functional size measurement (COSMIC function points) with quantum-native metrics (gate count, circuit width, qubit connectivity) for real-world Qiskit programs. It supports research in quantum software engineering, benchmarking, and machine learning.

This repository includes:
- Final dataset in `.csv` format
- Source programs in `python-sources/`

### Features

- Hundreds of real Qiskit programs from open-source repositories
- Multi-metric annotations for each file:
  - COSMIC function points (CFP)
  - Lines of Code (LOC)
  - Gate count
  - Circuit width
  - Qubit connectivity
- Open data format (CSV)
- Permissive MIT license

### Dataset structure

Columns in `Final_Dataset.csv`:
- File Name
- Events
- Total Entries
- Total Exits
- Total Writes
- Total Reads
- Total CFP
- LOC
- Gate Count
- Circuit Width
- Qubit Connectivity

### Quick start

```bash
git clone https://github.com/arefabacworkshop/QCFPDataSet.git
cd QCFPDataSet
```

```python
import pandas as pd
df = pd.read_csv("Final_Dataset.csv")
df.head()
```

### Contributing

We welcome contributions and feedback. Please open an [issue](https://github.com/arefabacworkshop/QCFPDataSet/issues) or submit a [pull request](https://github.com/arefabacworkshop/QCFPDataSet/pulls).

### Citation

If you use this dataset in your research, please cite:

```
@article{abbasi2026qcfpdataset,
  title={QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale},
  author={Abbasi, Aref and KheirKhah, Esmaeil},
  journal={Preprint},
  year={2026},
  url={https://github.com/arefabacworkshop/QCFPDataSet}
}
```

Alternative:

Abbasi, A., & KheirKhah, E. (2026). QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale. https://github.com/arefabacworkshop/QCFPDataSet

### License

MIT License. See the badge link above for details.

### Authors and acknowledgments

- Lead author: Aref Abbasi
- Supervisor and collaborator: Esmaeil KheirKhah
- Thanks to the open-source Qiskit community and all contributors

### Contact

For questions or collaboration, please open an issue or email Aref Abbasi at aref_abbasi@hotmail.com.
