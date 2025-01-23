import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import time
import pathlib
import pandas as pd

from experiments.nuScenes.helper import plot_vehicle_nice, plot_vehicle_mm
from experiments.nuScenes.plot_things import nusc_map

sys.path.append("/Users/thanh/Documents/code/UnlikelihoodMotionForecasting/trajectron")
# sys.path.append("/home/t/thanh/UnlikelihoodMotionForecasting/trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from tensorboardX import SummaryWriter
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import visualization

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str, default='VEHICLE')
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
parser.add_argument("--log_dir", help="what dir to save training information (i.e., saved models, logs, etc)", type=str)
parser.add_argument("--log_tag", help="tag for the log folder", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    # if 'PEDESTRIAN' in trajectron.pred_state:
    #     trajectron.pred_state.pop('PEDESTRIAN')
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    # Create the log and model directory if they're not present.
    random_id = str(np.random.randint(999)).zfill(
        3)  # avoid the same path when run multiple experiment at the same time
    model_dir = os.path.join(args.log_dir, '_'.join([args.log_tag,
                                                     time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()),
                                                     random_id]))

    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    log_writer = SummaryWriter(log_dir=model_dir)

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            timestep = np.array([2])
            for i, scene in enumerate(tqdm(scenes)):
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples=500)

                predictions_mm = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples=1,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)  # This will trigger grid sampling

                # Plot predicted timestep for a scene in map
                my_patch = (0, 0, 0, 0)
                layers = ['drivable_area',
                          'road_segment',
                          'lane',
                          'ped_crossing',
                          'walkway',
                          'stop_line',
                          'road_divider',
                          'lane_divider']
                fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.1,
                                                    render_egoposes_range=False)
                ax.plot([],
                        [],
                        'w--o', label='Ground Truth',
                        linewidth=3,
                        path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])

                plot_vehicle_nice(ax,
                                  predictions,
                                  scene.dt,
                                  max_hl=10,
                                  ph=ph,
                                  map=None)

                plot_vehicle_mm(ax,
                                predictions_mm,
                                scene.dt,
                                max_hl=10,
                                ph=ph,
                                map=None)
                leg = ax.legend(loc='upper right', fontsize=20, frameon=True)
                ax.axis('off')
                for lh in leg.legendHandles:
                    lh.set_alpha(.5)
                ax.get_legend().remove()
                ax.set_title(f"{scene.name}-t: {timestep[0]}")
                log_writer.add_figure('eval/prediction', fig, i)

