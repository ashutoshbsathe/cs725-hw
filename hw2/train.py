import os
import torch
import pytorch_lightning as pl

from utils import parse_args, load_model, load_datamodule, get_model_name

# A simple PyTorch Lightning training script
# Ideally, you should not need to modify this file

def main():
    args = parse_args()
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
        logger=pl.loggers.TensorBoardLogger(save_dir=litmodel_name),
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        callbacks=[
            # Save the model checkpoint
            pl.callbacks.ModelCheckpoint(
                dirpath=litmodel_name,
                filename='{epoch:06d}-{train_loss:.3f}-{valid_acc:.3f}',
                save_top_k=1,
                monitor='valid_acc',
                mode='max',
                save_last=True,
                every_n_epochs=1,
            )
        ]
    )
    trainer.fit(litmodel, datamodule)
    print(trainer.validate(litmodel, datamodule, ckpt_path='best'))

if __name__ == '__main__':
    main()
