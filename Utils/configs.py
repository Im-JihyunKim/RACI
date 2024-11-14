import os, json, copy, datetime, argparse

class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):
        
        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()
            
        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)
            
        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        parents = [
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
        ]
        
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args
        
        config = cls()
        parser.parse_args(namespace=config)
        
        return config
    
    @classmethod
    def form_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)
    
    def save(self, ckpt_dir):
        path = os.path.join(ckpt_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)
    
    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.backbone,
            f'seed_{self.seed}',
            self.hash
        )
        return ckpt
    
    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg
            
    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--data-dir', type=str, default='./Data/')
        parser.add_argument('--test-ratio', type=float, default=.2)
        parser.add_argument('--valid-ratio', type=float, default=.2)

        parser.add_argument('--task', type=str, default='ET', choices=('VM', 'ET'))
        parser.add_argument('--NP', type=str, default='NP', choices=('NP', 'N', 'P'))
        parser.add_argument('--encoding-type', type=str, default='Count', choices=('Target', 'Count', 'CatBoost', 'OneHot'))
        parser.add_argument('--eqp', type=str, default="eqp", choices=("no_eqp", "eqp"))
        parser.add_argument('--agg', type=str, default="mean", choices=("mean", "attn", "channel"))
        return parser
    
    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("ET Prediction Model", add_help=False)
        parser.add_argument('--backbone', type=str, default='CNN', choices=('DNN', 'RNN', 'GRU', 'LSTM', 'CNN', 'Transformer'))

        # Default hyperparameters
        parser.add_argument("--dropout-rate", type=float, default=.1)
        parser.add_argument("--emb-dim", type=int, default=128)
        parser.add_argument('--hidden-dim', type=int, default=64)

        # Transformer backbone hyperparameters
        parser.add_argument('--n-head', type=int, default=4,
                            help="the number of heads in the multiheadattention models")
        parser.add_argument('--dim-feedforward', type=int, default=32,
                            help="the dimension of the feedforward network model")
        parser.add_argument('--num-encoder-layers', type=int, default=3,
                            help="the number of sub-encoder-layers in the encoder")
        parser.add_argument("--feature-dim", type=int, default=4,
                            help="feature-dim must be divisible by n-head")
        
        # RNN-based backbone hyperparameters
        parser.add_argument('--bidirectional', type=bool, default=True)
        parser.add_argument('--num-layers', type=int, default=2)

        # 1D CNN backbone hyperparameters
        parser.add_argument("--kernel-size", type=int, default=2)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--dilation", type=int, default=1)
        return parser
    
    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size.')
        parser.add_argument('--num-workers', default=0, type=int)
        parser.add_argument("--epochs", default=1000, type=int)
        parser.add_argument("--warm-up", default=0, type=int, help='Number of iterations per epoch')
        
        parser.add_argument('--lr', type=float, default=5e-3, help='Base learning rate to start from.')
        parser.add_argument("--lr-scheduler", default="cosine", choices=(None, "step", "lambda", "cosine", "cosine_annealing", "cosine_annealing_warm"))
        
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=1e-3, type=float, help='l2 weight decay')
        parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam', 'rms_prop'))
        
        parser.add_argument("--loss", type=str, default="mse")
        parser.add_argument("--et-weight", type=str, default="same", choices=("same", "range", "var"))
        
        parser.add_argument('--patience', type=int, default=500)
        parser.add_argument('--gpus', type=int, nargs='+', default=0, help='')
        return parser
    
    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint-root', type=str, default='./Results/', help='Top-level directory of checkpoints.')
        return parser