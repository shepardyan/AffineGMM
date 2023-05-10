# Linear Transformations of Multivariate Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are finite mixture models. They are suitable for affine transformation-based probabilistic analysis, such as DC probabilistic power flow.

## Basic Usage

## TODOs

Sparse matrix support.

More APIs.

## Acknowledgement

The PDF functions of GMMs are vectorized using the code provided by [Gregory Gundersen](https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/). 

The fast approximation of CDF function of GMMs is from the [approxcdf package](https://github.com/david-cortes/approxcdf).

## Reference

[1] J. T. Flåm, “The Linear Model under Gaussian Mixture Inputs: Selected Problems in Communications,” Doctoral thesis, Norges teknisk-naturvitenskapelige universitet, Fakultet for informasjonsteknologi, matematikk og elektroteknikk, Institutt for elektronikk og telekommunikasjon, 2013. Accessed: Oct. 29, 2022. [Online]. Available: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2370728

[2] Z. Wang, C. Shen, F. Liu, and F. Gao, “Analytical Expressions for Joint Distributions in Probabilistic Load Flow,” IEEE Transactions on Power Systems, vol. 32, no. 3, pp. 2473–2474, May 2017, doi: 10.1109/TPWRS.2016.2612881.

