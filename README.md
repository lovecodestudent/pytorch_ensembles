# Ensembling PyTorch models

## Getting Started

Create an Ensemble object from src/ensemble.py

```
from src.ensemble import Ensemble

my_ensemble = Ensemble('/path/to/models/', ensemble_type='avg')

input = torch.randn(1, 3, 224, 224) # input tensor image -> (batch_size, channels, height, width)

prediction = my_ensemble.predict_ensemble(input)

# prediction stores the index of the predicted class.

```

## Author

[Prathamesh Mandke](mailto:pkmandke@vt.edu).
