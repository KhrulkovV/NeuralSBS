# NeuralSBS and SBS180K dataset
PyTorch Implementation of Neural Side-By-Side: Predicting Human Preferences for No-Reference Super-Resolution Evaluation
## SBS180K dataset
You can download the dataset from [this Dropbox url](https://www.dropbox.com/s/45tz3m5al9axyc5/NeuralSBS_dataset.zip).

## Pretrained model.
To build a model, use the following code.
```
from model import get_score_model
score_model = get_score_model('inception_v3', pretrained=True)
```
Checkpoint used for evaluation is available at [this Dropbox url](https://www.dropbox.com/s/gwalk982rombtov/neuralsbs.pth)
It can be loaded as 
```
score_model.load_state_dict(torch.load('neuralsbs.pth')['model_state_dict'])
score_model.eval()
```
