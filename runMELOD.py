import argparse
from src.train.trainer import Trainer
from src.train.config import config_override

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='MELOD')
    parser.add_argument('--embed_size', default=128, type=int, help='embedding size')
    parser.add_argument('--sas_prob', default=3, type=int, choices=[1, 2, 3, 4],
                        help='teach sample strategy, 1:[0.4, 0.3, 0.3], 2:[0.5, 0.3, 0.2], 3:[0.8, 0.1, 0.1], '
                             '4:[1., 0., 0.]')
    parser.add_argument('--DIIN_loss_type', default='MR', type=str, choices=['MR', 'BCE'],
                        help='loss type for DIIN, mr: margin rank loss, bec: binary cross entropy loss')
    parser.add_argument('--alpha', default=0.4, type=float, help='margin for DIIN loss')
    parser.add_argument('--lamda', default=0.2, type=float, help='weight for induction loss')
    parser.add_argument('--hit_range', default=100, type=int, help='valid hit range for seq reward')
    parser.add_argument('--sample_size', default=100, help='sample from top k item, [100, 200, -1:all]')
    parser.add_argument('--prob_sharpen', default=1., help='sharpen sample probability')
    parser.add_argument('--episode_num', default=1, type=int, help='episode number')

    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of transformer head')
    parser.add_argument('--ffn_hidden', type=int, default=512, help='transformer ffn hidden size')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='attention score dropout')
    parser.add_argument('--ffn_dropout', default=0.5, type=float, help='embedding dropout probability')
    parser.add_argument('--reward_agg', default='balance', type=str, choices=['balance', 'sum'])
    # Data
    parser.add_argument('--dataset', default='toys', type=str)
    parser.add_argument('--use_tar_len', action='store_false', help='if use target sequence')
    parser.add_argument('--target_len', default=3, type=int, help='ground-truth target length')
    # Training
    parser.add_argument('--session_len', default=3, type=int, help='next session length')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0, type=float, help='l2 normalization')  # 1e-06
    parser.add_argument('--patience', default=5, help='early stop patience')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--mark', default='')
    # Evaluation
    parser.add_argument('--train_batch', default=512, type=int)
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS_R@0.2', type=str, help='[LS, LS_R@0.x, PS]')
    parser.add_argument('--eval_mode', default='uni100', help="['uni100', 'pop100', 'full']")
    parser.add_argument('--k', default=[5, 10, 20], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@10')

    config = parser.parse_args()

    trainer = Trainer(config)
    trainer.start_training()


