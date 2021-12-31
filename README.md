# Cuisine_Prediction

![GitHub](https://img.shields.io/github/license/hiyouga/cuisine_prediction)

The technical report can be found [here](technical_report.pdf).

## Requirement

- Python >= 3.7
- torch >= 1.9.0
- numpy >= 1.17.2
- matplotlib >= 3.1.1 [optional]
- scikit-learn >= 0.21.3 [optional]

## Preparation

### Clone

```bash
git clone https://github.com/hiyouga/Cuisine_Prediction.git
```

### Create an anaconda environment:

```bash
conda create -n cuisine python=3.7
conda activate cuisine
pip install -r requirements.txt
```

## Usage

### Training

```sh
python main.py
```

### Ensemble

```sh
python main.py --ensemble [1.pt,2.pt,3.pt]
```

### Visualization with t-SNE

```sh
python main.py --checkpoint [*.pt]
```

The checkpoint files can be found in the `state_dict` folder.

### Show help message

```sh
python main.py -h
```

## Acknowledgements

This is a personal homework for "Machine Learning" in BUAA Graduate School.

## Contact

hiyouga [AT] buaa [DOT] edu [DOT] cn

## License

MIT
