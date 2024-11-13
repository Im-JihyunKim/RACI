import os, sys, rich, torch, time, random
import numpy as np

from Tasks.Trainer import Trainer
from Utils.logger import get_rich_logger
from Utils.configs import ConfigBase as configuration

import warnings
warnings.filterwarnings(action='ignore')

def main(local_rank:int):
    # Configuration
    configs = configuration.parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in [configs.gpus]])

    # Seed Fix
    np.random.seed(configs.seed)
    random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed_all(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make Checkpoint dir and Save configs
    rich.print(configs.__dict__)
    ckpt_dir = configs.checkpoint_dir
    configs.save(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set default gpus number
    torch.cuda.set_device(local_rank)

    # Define logger
    logfile = os.path.join(ckpt_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    logger.info(f'Checkpoint directory: {ckpt_dir}')

    # Model Training and Evaluation
    rich.print(f"Training Start")
    start = time.time()

    trainer = Trainer(
                    configs=configs,
                    device=local_rank,
                    ckpt_dir=ckpt_dir
                    )
    
    trainer.run(logger=logger)

    end_sec = time.time() - start

    if logger is not None:
        end_min = end_sec / 60
        logger.info(f"Total Training Time: {end_min: 2f} minutes")
        logger.handlers.clear()

if __name__ == '__main__':
    try:
        main(0)

    except KeyboardInterrupt:
        sys.exit(0)