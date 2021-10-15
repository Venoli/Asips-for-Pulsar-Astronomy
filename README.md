
<p><img src="https://github.com/Venoli/Asips-for-Pulsar-Astronomy/blob/AddGaussianHellingerSplitCriterion/docs/_static/images/Asips-logo.png?raw=true" height="100"/></p>
<h1>Asips for Pulsar Astronomy</h1>  
'Asips' is a Research cunducted for automating pulsar candidate selection. This is the API of Asips which can be used by anyone.
This implementation uses the HTRU2 dataset.

### Quick links
* [Website (Demo web app)](https://asips-for-pulsars-astronomy.web.app/)
* [Documentation](https://github.com/Venoli/Asips-for-Pulsar-Astronomy#readme)

# :fire: Features

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


## 🛠 Installation
Clone the repository

```python
git clone https://github.com/Venoli/Asips-for-Pulsar-Astronomy.git
```
Run the Jupyter notebook inside the src folder
```python
cd src/asips
!python flask_api.py
```

## ⚡️ Quick Start

Since this is developed using Flask, the above code will start the server on http://localhost:5000/. (This will refer as BASE_URL in the below sections)

### Pretrain
Below request will pretrain the model. <br>
count: pretrain count
```python
BASE_URL/pretrain/<count>
```

### Predict
Below request will make a prediction using the model.  <br>
count: number of samples to predict
```python
BASE_URL/predict/<count>
```

### Learn from all
By below request model will learn from all of the early predictions
```python
BASE_URL/learn-from-all
```
### Learn by id
By below request model will learn from sample with given id.  <br>
id: id of the sample
```python
BASE_URL/learn/<id>
```

### Test with Another Classifier
By below request previouse predictions can be verified using another model.  <br>
model: name of the model. 
      (smoteBagging, underOverBagging)
```python
BASE_URL/test-with-other-classifier/<model>
```

# :open_book: Credits
- Extremely Fast Decission Tree (EFDT) [1] - GH-EFDT is a improved version of EFDT
- Hellinger Distance among Gaussian Distributions [2] - The improvemrnt done by using hellinger distance
- Scikit-Multiflow [3] - research, implimentation and testing was done on top of the scikit-multiflow library.
  scikit-multiflow implementation of EFDT was modified.
- Gaussian Hellinger Very Fast Decision Tree [4] - Main encouragement behind the GH-EFDT
- HTRU2 dataset [5] - The dataset that used in development

[1] C. Manapragada, G. I. Webb, and M. Salehi, “Extremely Fast Decision Tree,” 2018. DOI: 10.1145/nnnnnnn. arXiv: 1802.08780v1.

[2] R. J. Lyon, J. M. Brooke, J. D. Knowles, and B. W. Stappers, “Hellinger distance trees for imbalanced streams,” in Proceedings - International Conference on         Pattern Recognition, Institute of Electrical and Electron- ics Engineers Inc., Dec. 2014, pp. 1969–1974, ISBN: 9781479952083. DOI: 10.1109/ICPR.2014.344.           arXiv: 1405.2278.

[3] Montiel, J., Read, J., Bifet, A., & Abdessalem, T. (2018). Scikit-multiflow: A multi-output streaming framework. The Journal of Machine Learning Research,           19(72):1−5.

[4] R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, and J. D. Knowles, “Fifty years of pulsar candidate selection: From simple filters to a new principled         real- time classification approach,” Monthly Notices of the Royal Astronomical Society, vol. 459, no. 1, pp. 1104– 1123, Jun. 2016, ISSN: 13652966. DOI:             10.1093/mnras/ stw656. arXiv: 1603.05166.

[5] R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time     classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
