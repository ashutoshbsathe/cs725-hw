import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from utils import parse_visualization_args, load_model, get_model_name, gen_meshgrid

def main():
    args = parse_visualization_args()
    print(args)
    args.dataset = 'binary'
    args.model = 'logistic_regression'

    # Create `log_dir` if it doesn't exist
    os.makedirs(f'{args.log_dir}/{args.dataset}/', exist_ok=True)

    # Load the dataset
    train_x = np.load(f'./data/{args.dataset}/train_x.npy')
    train_y = np.load(f'./data/{args.dataset}/train_y.npy')
    print(f'Loaded training dataset\nInput(x) shape = {train_x.shape}, Target(y) shape = {train_y.shape}')

    # Prepare the model
    model = load_model(args.model)()
    model_name = get_model_name(args)

    # Preprocess the data
    train_x = model.preprocess(train_x)

    # Visualization trackers
    train_losses = []
    train_accs = []
    x1, x2, eval_x = gen_meshgrid(args.grid_size, train_x, args.epsilon)
    pred_eval_y = []

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
        train_y_pred = model.get_prediction(train_x)
        train_acc = (train_y_pred == train_y).mean()
        
        pred_eval_y.append(model.get_prediction(eval_x))
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        pbar.set_description(f'train_loss={train_loss:.2f}, train_acc={train_acc * 100:.2f}%')

    print(f'==== Training completed. ====')
    
    fig = plt.figure()
    ax = plt.gca()

    def draw_boundary(i):
        pred_y = pred_eval_y[i].reshape((args.grid_size, args.grid_size))
        ax.clear()
        ax.set_xlim(x1.min(), x1.max())
        ax.set_ylim(x2.min(), x2.max())
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Epoch {i+1:04d}: Train loss = {train_losses[i]:.3f}. Train accuracy = {train_accs[i]*100:.2f}%.')
        ax.contourf(x1, x2, pred_y, cmap=args.cmap, alpha=args.contourf_alpha)
        ax.contour(x1, x2, pred_y, colors='k', linewidths=args.contour_linewidth)
        ax.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=args.cmap)
    
    anim = animation.FuncAnimation(fig, draw_boundary, len(pred_eval_y), interval=50, blit=False)
    
    try:
        writer = animation.PillowWriter(
            fps=args.gif_fps,
            metadata=dict(
                artist='CS725-2023 HW1: Ashutosh Sathe and Krishnakant Bhatt'
            ),
            bitrate=args.gif_bitrate,
        )

        def print_anim_progress(i, n):
            msg = 'Starting GIF creation' if i == n else f'Rendering frame {i}/{n}'
            print(msg, end='\r', flush=True)

        fname = model_name + '.anim.gif'
        anim.save(fname, writer=writer, dpi=args.gif_dpi, progress_callback=print_anim_progress)
        print(f'Animation saved at: {fname}')
    except Exception as e:
        print(f'Unable to write GIF file. You may need `ffmpeg`\nError={repr(e)}')
        plt.show()

if __name__ == '__main__':
    main()