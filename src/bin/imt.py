import argparse
from src.main import interactive_FBS

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str,
                    help="""Name of the model.""")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--ref_path", type=str,
                    help="""Path to ref file.""")

parser.add_argument("--fw_model_path", type=str,
                    help="""Path to forward model files.""")

parser.add_argument("--bw_model_path", type=str,
                    help="""Path to backward model files.""")

parser.add_argument("--config_path", type=str,
                    help="""Path to config file.""")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--beam_size", type=int, default=5,
                    help="""Beam size.""")

parser.add_argument("--saveto", type=str,
                    help="""Result prefix.""")

parser.add_argument("--keep_n", type=int, default=-1,
                    help="""To keep how many results. This number should not exceed beam size.""")

parser.add_argument("--use_gpu", action="store_true")

parser.add_argument("--max_steps", type=int, default=150,
                    help="""Max steps of decoding. Default is 150.""")

parser.add_argument("--alpha", type=float, default=-1.0,
                    help="""Factor to do length penalty. Negative value means close length penalty.""")

parser.add_argument("--try_times", type=int, default=1,
                    help="""times to try revise per sentence""")

parser.add_argument("--imt_step", type=int, default=1,
                    help="""revise step""")

parser.add_argument("--bidirection", action="store_true", default=False)

parser.add_argument("--online_learning", action="store_true", default=False)


def run(**kwargs):

    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    interactive_FBS(args)

if __name__ == '__main__':
    run()