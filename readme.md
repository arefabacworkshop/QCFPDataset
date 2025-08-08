



````markdown

\# QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale



!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)



---



\## Overview



\*\*QCFPDataSet\*\* is the first large-scale, open-access dataset that combines classical functional size measurement (COSMIC function points) with quantum-native metrics (gate count, circuit width, qubit connectivity) for real-world Qiskit programs. This project advances research in quantum software engineering, benchmarking, and machine learning. The dataset and tools are provided for free to support academic and industrial research, education, and open science.



This repository includes:

\- The complete dataset in `.csv` format

\- Python scripts for automated measurement and analysis

\- Example Jupyter notebooks and documentation



---



\## Features



\- \*\*1,000+ real Qiskit programs\*\* from open-source GitHub repositories

\- \*\*Multi-metric annotations\*\* for each file:

&nbsp;   - COSMIC function points (CFP)

&nbsp;   - Lines of Code (LOC)

&nbsp;   - Gate count

&nbsp;   - Circuit width

&nbsp;   - Qubit connectivity

\- \*\*Automated, reproducible measurement pipeline\*\* for static analysis

\- \*\*Open data formats\*\* (CSV, JSON)

\- \*\*Permissive MIT license\*\*



---



\## Dataset Structure



Each dataset entry includes:

\- \*\*Filename and repository information\*\*

\- \*\*Classical metrics\*\*: COSMIC Function Points, LOC

\- \*\*Quantum-native metrics\*\*: Gate count, circuit width, qubit connectivity

\- \*\*COSMIC breakdown\*\*: Entries (E), Exits (X), Reads (R), Writes (W)

\- \*\*Metadata and quality control flags\*\*





---



\## How to Use



1\. \*\*Download the dataset\*\* from the repository.

2\. \*\*Cite this dataset\*\* if you use it in your work (see below).



---



\## Getting Started



```bash

git clone https://github.com/arefabacworkshop/QCFPDataSet.git

cd QCFPDataSet

pip install -r requirements.txt

\# Example: run analysis pipeline

python pipeline/analyze\_qcfp\_dataset.py

````



---



\## Contributing



We welcome contributions and feedback!

To suggest changes, please open an \[issue](https://github.com/arefabacworkshop/QCFPDataSet/issues) or submit a \[pull request](https://github.com/arefabacworkshop/QCFPDataSet/pulls).



---



\## Citing QCFPDataSet



If you use this dataset in your research, \*\*please cite\*\*:



```

@article{abbasi2026qcfpdataset,

&nbsp; title={QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale},

&nbsp; author={Aref Abbasi and Esmaeil KheirKhah},

&nbsp; journal={Preprint},

&nbsp; year={2026},

&nbsp; url={https://github.com/arefabacworkshop/QCFPDataSet}

}

```



Or:



> Abbasi, A., \& KheirKhah, E. (2026). QCFPDataSet: Automated Functional and Quantum-Native Software Measurement at Scale. \[https://github.com/arefabacworkshop/QCFPDataSet](https://github.com/arefabacworkshop/QCFPDataSet)



---



\## License



This project is licensed under the \[MIT License](LICENSE).



---



\## Authors \& Acknowledgments



\*\*Lead Author:\*\*

Aref Abbasi (\[GitHub](https://github.com/yourusername))



\* Concept, implementation, data collection, analysis, and manuscript writing



\*\*Supervisor and Collaborator:\*\*

Esmaeil KheirKhah (\[GitHub](https://github.com/theirusername))



\* Research supervision, project guidance, manuscript review



Special thanks to the open-source Qiskit community and all contributors.



---



\*\*Contact:\*\*

For questions or collaboration, please open an issue or email Aref Abbasi at \[your.email@example.com](mailto:aref\_abbasi@hotmail.com).



---



\*\*Enjoy exploring QCFPDataSet and please remember to cite our work if you use it!\*\*



```



---



