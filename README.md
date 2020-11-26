# MIML: A Java library for MIML learning
* [What is MIML library?](https://github.com/kdis-lab/MIML/blob/master/README.md#what-is-miml-library)
* [Installation, tutorials and documentation](https://github.com/kdis-lab/MIML/blob/master/README.md#installation-tutorials-and-documentation)
* [Methods included](https://github.com/kdis-lab/MIML/blob/master/README.md#methods-included)
* [References](https://github.com/kdis-lab/MIML/blob/master/README.md#references)
* [Citation](https://github.com/kdis-lab/MIML/blob/master/README.md#citation)
* [License](https://github.com/kdis-lab/MIML/blob/master/README.md#license)

## What is MIML library?
MIML is a modular Java library whose aim is to ease the development, testing and comparison of classification algorithms for multi-instance multi-label learning (MIML). It includes three different approaches for solving a MIML problem: transforming the problem to multi-instance, transforming the problem to multi-label problem, and solving directly the MIML problem. Besides, it provides holdout and cross-validation procedures, standard metrics for performance evaluation as well as report generation. Algorithms can be executed by means of *xml* configuration files. It is platform-independent, extensible, free and open source.

## Installation, tutorials and documentation
The documentattion can be found in the doc folder and includes:
* Installation guide
* Tutorials
* A guide about ....
* An example about how to add a new classification method to MIML


## Methods included

|  MIML to MI problem  |                     |
|:--------------------:|---------------------|
| Label Transformation | MI Algorithm (Weka) |
|                      | CitationKNN         |
|                      | MDD                 |
|                      | MIDD                |
|                      | MIBoost             |
|                      | MILR                |
|                      | MIOptimalBall       |
|          BR          | MIRI                |
|                      | MISMO               |
|                      | MISVM               |
|                      | MITI                |
|                      | MIWrapper           |
|                      | SimpleMI            |
|                      | CitationKNN         |
|          LP          | MIWrapper           |
|                      | SimpleMI            |


|  MIML to ML problem  |                     |
|:--------------------:|---------------------|
| Bag Transformation | ML Algorithm (Mulan) |
|                      | BR        |
|                      | LP                 |
|                      | RPC                |
|                      | CLR             |
|                      | BRkNN                |
|          Arithmetic            | DMLkNN       |
|          Geometric          | IBLR                |
|          Min-Max            | MLkNN               |
|                      | HOMER               |
|                      | RAkEL               |
|                      | PS           |
|                      | EPS            |
|                      | CC         |
|                    | ECC           |
|                      | MLStacking            |


| MIML method |  |
| ------------- | ------------- |
| Bagging  | Contenido de la celda  |
| MIMLkNN  | Contenido de la celda  |
| MLkNN  | Contenido de la celda  |



## References

## Citation
This work has been performed by A. Belmonte, A. Zafra and E. Gibaja and is currently in a reviewing process.

## License
MIML library is released under the GNU General Public License [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).
