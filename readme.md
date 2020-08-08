# TextRay
Project code for ACM MM2020 paper: "TextRay: Contour-based Geometric Modeling for Arbitrary-shaped Scene Text Detection"

# Dependencies

- PyTorch
- MMDetection

```
pip install -r requirements.txt # install dependencies
git submodule update --init # clone mmdetection
cd mmdetection
python setup.py develop # compile mmdetection
cd ../curve/ops/ && sh ./compile.sh # compile custom operations
```

# Notes

1, To train a model:

```
cd experiments/CTW_cheby
./train.sh 0,1,2,3 # training on 4 gpus
```

Models will be saved in `./work_dirs`.

2, To test a model:

```
cd experiments/CTW_cheby
./test.sh 0,1,2,3 # test on 4 gpus
```

A `.pkl` file containing detection results is generated.

Evaluation:
Replace the `.pkl` filename in `curve/notebooks/ctw_eval.ipynb`.

The training order is `Prertrain_CTW`-->`CTW_cheby` and `Pretrain_Total`-->`Total_cheby`.

Models are available in the following links:
Google Drive: https://drive.google.com/drive/folders/1nhZuF_S6yvl57RZuC2IgACBqK26sZK5c?usp=sharing
Baidu Disk: https://pan.baidu.com/s/1J7fRtKrFdUFvq3eu1P9RVQ (password: cf6b)
