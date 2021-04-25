
<p align="center"><img src="https://github.com/Venoli/Asips-for-Pulsar-Astronomy/blob/AddGaussianHellingerSplitCriterion/docs/_static/images/Asips-logo.png?raw=true" height="100"/></p>
<h1>Asips for Pulsar Astronomy</h1>  
'Asips' is a Research cunducted for automating pulsar candidate selection. This is the API of Asips which can be used by anyone


### Quick links
* [Webpage](https://asips-for-pulsars-astronomy.web.app/)
* [Documentation](https://github.com/Venoli/Asips-for-Pulsar-Astronomy/edit/AddGaussianHellingerSplitCriterion/README.md)

# Features

### Gaussian Hellinger Extremely Fast Decision Tree (GH-EFDT)
This is the main output of the research. GH-EFDT is a stream learning algorithm. This is an 
improviced version of Extremely Fast Decision Tree [1] for imbalanced streams. Hellinger Distance amoung 
Gaussian destributions were used as a split criterion [2] when handling the imbalanced problem.
This algorithm is suitable for candidate selection since it is,

- 1) Accurate
- 2) Not biased toward the majority class
- 3) Learn incrementally
- 4) Not Harmed by concept drift
- 5) Fast


### Other classification Algorithms
This API provide verification feature from other libraries. For now this provides,

- 1) OnlineSMOTEBaggingClassifier
- 2) OnlineUnderOverBaggingClassifier


## Quick Start
```python
%matplotlib notebook
```


# Credits
- Extremely Fast Decission Tree (EFDT) [1] - GH-EFDT is a improved version of EFDT
- Hellinger Distance among Gaussian Distributions [2] - The improvemrnt done by using hellinger distance
- Scikit-Multiflow [3] - research, implimentation and testing was done on top of the scikit-multiflow library.
  scikit-multiflow implementation of EFDT was modified.
- Gaussian Hellinger Very Fast Decision Tree [4] - Main encouragement behind the GH-EFDT

[1] C. Manapragada, G. I. Webb, and M. Salehi, “Extremely Fast Decision Tree,” 2018. DOI: 10.1145/nnnnnnn. arXiv: 1802.08780v1.
[2] R. J. Lyon, J. M. Brooke, J. D. Knowles, and B. W. Stappers, “Hellinger distance trees for imbalanced streams,” in Proceedings - International Conference on         Pattern Recognition, Institute of Electrical and Electron- ics Engineers Inc., Dec. 2014, pp. 1969–1974, ISBN: 9781479952083. DOI: 10.1109/ICPR.2014.344.           arXiv: 1405.2278.
[3] Montiel, J., Read, J., Bifet, A., & Abdessalem, T. (2018). Scikit-multiflow: A multi-output streaming framework. The Journal of Machine Learning Research,           19(72):1−5.
[4] R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, and J. D. Knowles, “Fifty years of pulsar candidate selection: From simple filters to a new principled         real- time classification approach,” Monthly Notices of the Royal Astronomical Society, vol. 459, no. 1, pp. 1104– 1123, Jun. 2016, ISSN: 13652966. DOI:             10.1093/mnras/ stw656. arXiv: 1603.05166.



