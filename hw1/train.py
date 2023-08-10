import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import parse_args, load_model, get_model_name

def main():
    args = parse_args()
    print(args)

    # Create `log_dir` if it doesn't exist
    os.makedirs(f'{args.log_dir}/{args.dataset}/', exist_ok=True)

    # Load the dataset
    train_x = np.load(f'./data/{args.dataset}/train_x.npy')
    train_y = np.load(f'./data/{args.dataset}/train_y.npy')

    valid_x = np.load(f'./data/{args.dataset}/val_x.npy')
    valid_y = np.load(f'./data/{args.dataset}/val_y.npy')

    print(f'Loaded training dataset\nInput(x) shape = {train_x.shape}, Target(y) shape = {train_y.shape}')
    print(f'Loaded validation dataset\nInput(x) shape = {valid_x.shape}, Target(y) shape = {valid_y.shape}')

    # Prepare the model
    model = load_model(args.model)()
    model_name = get_model_name(args)
    best_valid_acc = float('-inf')
    best_acc_epoch = None

    # Preprocess the data
    train_x = model.preprocess(train_x)
    valid_x = model.preprocess(valid_x) 
    """
    Note: ideally we should be using transform parameters from training data itself, 
    but here the datasets are so small that it doesn't significantly affect the performance
    """

    # Visualization trackers
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    print('==== Training ====')
    pbar = tqdm(range(args.num_epochs))
    for e in pbar:
        # calculate gradient. ensure grad.shape == model.weights.shape
        grad = model.calculate_gradient(train_x, train_y)
        assert grad.shape == model.weights.shape, f'Shape mismatch for gradient and weights. Gradient shape = {grad.shape}. Weights shape = {model.weights.shape}'
        # update the params
        model.update_weights(grad, args.learning_rate, args.momentum)

        # weight update completed, calculate loss/accuracy on train and validation splits
        train_loss = model.calculate_loss(train_x, train_y)
        valid_loss = model.calculate_loss(valid_x, valid_y)

        train_y_pred = model.get_prediction(train_x)
        train_acc = (train_y_pred == train_y).mean()
        valid_y_pred = model.get_prediction(valid_x)
        valid_acc = (valid_y_pred == valid_y).mean()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        if valid_acc >= best_valid_acc:
            # save the model weights
            np.save(model_name + '.weights.npy', model.weights)
            best_valid_acc = valid_acc
            best_acc_epoch = e+1
            print(f'Saved weights at epoch {e+1}, valid_acc = {valid_acc}, best_valid_acc = {best_valid_acc}')
        
        pbar.set_description(f'train_loss={train_loss:.2f}, valid_loss={valid_loss:.2f}, valid_acc={valid_acc:.2f}')

    print(f'==== Training completed. best_valid_acc = {best_valid_acc * 100:.2f}% obtained at epoch {best_acc_epoch}. ====')

    # Save training plot
    plt.clf()
    plt.plot(range(1, args.num_epochs+1), train_losses, label='Train loss')
    plt.plot(range(1, args.num_epochs+1), valid_losses, label='Validation loss')
    plt.axvline(best_acc_epoch, label=f'Best epoch({best_acc_epoch})', color='red', linestyle='--')
    plt.xlim(0, args.num_epochs+1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name + '.lossplot.pdf', bbox_inches='tight')

    # Save accuracies plot
    plt.clf()
    plt.plot(range(1, args.num_epochs+1), train_accs, label='Train accuracy')
    plt.plot(range(1, args.num_epochs+1), valid_accs, label='Validation accuracy')
    plt.axvline(best_acc_epoch, label=f'Best epoch({best_acc_epoch})', color='red', linestyle='--')
    plt.xlim(0, args.num_epochs+1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(model_name + '.accplot.pdf', bbox_inches='tight')

    # Save losses/accuracies 
    with open(model_name + '.train_losses.txt', 'w') as f:
        f.write('\n'.join(str(loss) for loss in train_losses))
    with open(model_name + '.valid_losses.txt', 'w') as f:
        f.write('\n'.join(str(loss) for loss in valid_losses))
    with open(model_name + '.train_accs.txt', 'w') as f:
        f.write('\n'.join(str(acc) for acc in train_accs))
    with open(model_name + '.valid_accs.txt', 'w') as f:
        f.write('\n'.join(str(acc) for acc in valid_accs))
    
    print(f'Logs and plots for this run written at {args.log_dir}/{args.dataset}/')

if __name__ == '__main__':
    main()