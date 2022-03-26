import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import ConvTasNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", type=str, required=True, help="Test directory including the csv files")
parser.add_argument("--task", type=str,required=True, help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`")
parser.add_argument("--model_path", type=str,required=False, default='', help="Path to the model (either best model or checkpoint")
parser.add_argument("--result_dir", type=str, required=False, default='', help="Directory where the eval results will be stored")
parser.add_argument("--from_checkpoint", type=bool, default=False, help="Model in model path is checkpoint, not final model. Default: 0")
parser.add_argument("-w", "--write_wav", type=bool, default=False, help="Wrtite inferred wav or not")
parser.add_argument("--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model")

COMPUTE_METRICS = ["si_sdr", "sdr", "stoi", "pesq"]

def update_compute_metrics(compute_wer, metric_list):
    if not compute_wer:
        return metric_list
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ModuleNotFoundError:
        import warnings

        warnings.warn("Couldn't find espnet installation. Continuing without.")
        return metric_list
    return metric_list + ["wer"]


def main(conf):
    
    compute_metrics = COMPUTE_METRICS
    if conf['model_path'] == '':
       conf['model_path'] = os.path.join(conf['exp_dir'], 'best_model.pth')
    if conf['result_dir'] == '':
        conf['result_dir'] = os.path.join(conf['exp_dir'], conf['exp_dir'].split('/')[-1]+'.yml')
    
    if not conf["from_checkpoint"]:
        print('>> 加载模型文件 '+conf['model_path'])
        model = ConvTasNet.from_pretrained(conf['model_path'])
    else:
        raise NotImplemented
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    
    print('>> 读取test_set...')
    test_set = LibriMix(
        csv_dir=conf["csv_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = conf["result_dir"]
    if not os.path.exists(eval_save_dir):
        os.makedirs(eval_save_dir, exist_ok=True)
    ex_save_dir = os.path.join(eval_save_dir, "wav_est/")
    if conf['write_wav']:
        os.makedirs(ex_save_dir, exist_ok=True)
    print(f'>> 结果将保存到 {eval_save_dir}')
    
    print('')
    print('### START EVALUATION###')
    series_list = []
    torch.no_grad().__enter__()
    sisdr_i_sum = 0
    with tqdm(total=len(test_set)) as pbar:
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, ids = test_set[idx]
            mix, sources = tensors_to_device([mix, sources], device=model_device)
            est_sources = model(mix.unsqueeze(0))
            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )
            utt_metrics["mix_path"] = test_set.mixture_path
            est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
            series_list.append(pd.Series(utt_metrics))

            if conf['write_wav']:
                # Save some examples in a folder. Wav files and metrics as text.
                if True:
                    mixture_name = test_set.df.iloc[idx]['mixture_path'].split('/')[-1].split('.')[0]
                    # Loop over the sources and estimates
                    for src_idx, est_src in enumerate(est_sources_np_normalized):
                        sf.write(
                            ex_save_dir + f"{mixture_name}_est{src_idx}.wav",
                            est_src,
                            conf["sample_rate"],
                        )

            sisdr_i_sum += np.round(utt_metrics['si_sdr']-utt_metrics['input_si_sdr'], 2)
            
            pbar.set_postfix(sisdr_i=sisdr_i_sum/(idx+1))
            pbar.update(1)
            
    # Save all metrics to the experiment folder.
    print('>> 保存结果...')
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    print('>> 打印结果...')
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print('')
    print(f'>> 被测模型地址: {conf["model_path"]}')
    print("Overall metrics :")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(conf['model_path'], map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict,
        metrics=final_results,
        train_conf=train_conf,
    )

    print('')
    print('### EVAL COMPLETED ###')
    print('')
    from IPython import embed
    embed()
    
    torch.cuda.empty_cache()
    print('')
    print('done.')

if __name__ == "__main__":
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    filelist = os.listdir(args.exp_dir)
    for file in filelist:
        if file.split('.')[-1] == 'yml':
            conf_path = os.path.join(args.exp_dir, file)
            break
    print('')
    print(f'>> 解析yml {conf_path}')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)