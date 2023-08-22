import torch
import numpy as np
from utils import load_model

def evaluate_model(dataset_name, model_name):
    print(f'Loading validation split of "{dataset_name}" dataset')
    valid_x = torch.Tensor(np.load(f'./data/{dataset_name}/valid_x.npy'))
    valid_y = torch.LongTensor(np.load(f'./data/{dataset_name}/valid_y.npy'))

    print(f'Loading "{model_name}" model')
    model = load_model(model_name)

    print(f'Loading weights')
    model = model.load_from_checkpoint(f'./submission/best_{model_name}.ckpt', map_location='cpu', lr=0)
    model.eval()

    with torch.no_grad():
        valid_x, valid_y = model.transform_input((valid_x, valid_y)) 
        out = model.validation_step((valid_x, valid_y))

    print(f'Validation loss = {out["valid_loss"]}. Validation accuracy = {out["valid_acc"] * 100:.2f}%.')

def main():
    print(32*'-')
    evaluate_model('simple', 'simple')
    print(32*'-')
    evaluate_model('digits', 'digits')
    print(32*'-')

if __name__ == '__main__':
    main()
