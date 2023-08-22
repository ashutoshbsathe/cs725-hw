import torch
import sys
import numpy as np
from utils import load_model

# Usage: `python make_kaggle_submission.py <path-to-best-digits-ckpt>`

model = load_model('digits').load_from_checkpoint(sys.argv[1], map_location='cpu', lr=0)
valid_x = torch.Tensor(np.load('./data/digits/valid_x.npy'))
valid_x, _ = model.transform_input((valid_x, torch.zeros(valid_x.size(0))))
y_pred = torch.zeros(valid_x.size(0)) # model.predict(valid_x)

with open('kaggle_upload.csv', 'w') as f:
    f.write('unique_id, y\n')
    f.write('\n'.join(f'{i}, {str(x.item())}' for i, x in enumerate(y_pred.long())))
