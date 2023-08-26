import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import parse_visualization_args, load_model, load_datamodule, get_model_name, gen_meshgrid

# A simple PyTorch Lightning trainer with a custom visualization callback
# Ideally, you should not need to modify this file ever

class VisualizationCallback(pl.Callback):
    def __init__(self, visualization_args, xform):
        self.visualization_args = visualization_args
        self.xform = xform
        self.pred_eval_y = []
        self.train_losses = []
        self.train_accs = []
        self.litmodel_name = get_model_name(visualization_args)
    
    def on_fit_start(self, trainer, pl_module):
        train_x = torch.Tensor(np.load('./data/simple/train_x.npy'))
        train_y = torch.LongTensor(np.load('./data/simple/train_y.npy'))
        self.train_x, self.train_y = self.xform((train_x, train_y))
        self.x1, self.x2, self.eval_x = gen_meshgrid(self.visualization_args.grid_size, self.train_x.numpy(), self.visualization_args.epsilon)
        self.eval_x = torch.Tensor(self.eval_x)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            out = pl_module.validation_step((self.train_x, self.train_y))
            self.train_losses.append(out['valid_loss'])
            self.train_accs.append(out['valid_acc'])
            self.pred_eval_y.append(pl_module.predict(self.eval_x).cpu().numpy())

    def on_fit_end(self, trainer, pl_module):
        fig = plt.figure()
        ax = plt.gca()
        matplotlib.colors.Normalize(vmin=0, vmax=3)

        def draw_boundary(i):
            pred_y = self.pred_eval_y[i].reshape((self.visualization_args.grid_size, self.visualization_args.grid_size))
            ax.clear()
            ax.set_xlim(self.x1.min(), self.x1.max())
            ax.set_ylim(self.x2.min(), self.x2.max())
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f'Epoch {i+1:04d}: Train loss = {self.train_losses[i]:.3f}. Train accuracy = {self.train_accs[i]*100:.2f}%.')
            ax.contourf(self.x1, self.x2, pred_y, cmap=self.visualization_args.cmap, alpha=self.visualization_args.contourf_alpha)
            ax.contour(self.x1, self.x2, pred_y, colors='k', linewidths=self.visualization_args.contour_linewidth)
            ax.scatter(self.train_x[:, 0], self.train_x[:, 1], c=self.train_y, cmap=self.visualization_args.cmap)
        
        anim = animation.FuncAnimation(fig, draw_boundary, len(self.pred_eval_y), interval=50, blit=False)
        
        try:
            writer = animation.PillowWriter(
                fps=self.visualization_args.gif_fps,
                metadata=dict(
                    artist='CS725-2023 HW1: Ashutosh Sathe and Krishnakant Bhatt'
                ),
                bitrate=self.visualization_args.gif_bitrate,
            )

            def print_anim_progress(i, n):
                msg = 'Starting GIF creation' if i == n else f'Rendering frame {i}/{n}'
                print(msg, end='\r', flush=True)

            fname = self.litmodel_name + '.training.anim.gif'
            anim.save(fname, writer=writer, dpi=self.visualization_args.gif_dpi, progress_callback=print_anim_progress)
            print(f'Animation saved at: {fname}')
        except Exception as e:
            print(f'Unable to write GIF file. You may need `ffmpeg`\nError={repr(e)}')
            plt.show()

def main():
    args = parse_visualization_args()
    args.model = 'simple'
    args.dataset = 'simple'
    print(args)

    # Create `log_dir` if it doesn't exist
    os.makedirs(f'{args.log_dir}/{args.dataset}/', exist_ok=True)

    # Seed everything for better reproducibility
    pl.seed_everything(args.seed)

    # Prepare the model
    litmodel = load_model(args.model)(args.learning_rate)
    litmodel_name = get_model_name(args)
    
    datamodule = load_datamodule(args.dataset)(batch_size=args.batch_size)

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=args.num_epochs,
        enable_checkpointing=False,
        logger=False,
        callbacks=[VisualizationCallback(args, litmodel.transform_input)]
    )
    trainer.fit(litmodel, datamodule)

if __name__ == '__main__':
    main()
