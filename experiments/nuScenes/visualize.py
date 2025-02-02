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

sys.path.append("/home/t/thanh/UnlikelihoodMotionForecasting/trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from tensorboardX import SummaryWriter
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


def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    if 'PEDESTRIAN' in trajectron.pred_state:
        trajectron.pred_state.pop('PEDESTRIAN')
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
            ############### MOST LIKELY Z ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])

            print("-- Evaluating GMM Z Mode (Most Likely)")
            for i, scene in enumerate(tqdm(scenes)):
                # timesteps = np.arange(scene.timesteps)
                # timesteps = scene.sample_timesteps(1, min_future_timesteps=ph)
                timesteps = np.array([0])  # TODO: revert back

                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)  # This will trigger grid sampling

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False,
                                                                       kde=False)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))

                # Plot predicted timestep for current scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timesteps[0]}")
                log_writer.add_figure('eval/prediction_most_likely', fig, i)

            print('ade {}'.format(np.mean(eval_ade_batch_errors)))
            print('fde {}'.format(np.mean(eval_fde_batch_errors)))

            ############### FULL ###############
            num_sample = 200
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            eval_road_viols = np.array([])
            check_result_total = []
            print("-- Evaluating Full")
            for i, scene in enumerate(tqdm(scenes)):
                # timesteps = np.arange(scene.timesteps)
                # timesteps = scene.sample_timesteps(1, min_future_timesteps=ph)
                timesteps = np.array([0]) # TODO: revert back

                predictions, check_result = eval_stg.predict(scene,
                                                             timesteps,
                                                             ph,
                                                             num_samples=num_sample,
                                                             min_future_timesteps=8,
                                                             z_mode=False,
                                                             gmm_mode=False,
                                                             full_dist=False,
                                                             check_checker=True)

                if not predictions:
                    continue

                prediction_dict, histories_dict, futures_dict = \
                    utils.prediction_output_to_trajectories(predictions,
                                                            scene.dt,
                                                            max_hl,
                                                            ph,
                                                            prune_ph_to_future=False)

                eval_road_viols_batch = []
                for t in prediction_dict.keys():
                    for node in prediction_dict[t].keys():
                        if node.type == args.node_type:
                            viols = compute_road_violations(prediction_dict[t][node],
                                                            scene.map[args.node_type],
                                                            channel=0)
                            if viols == num_sample:
                                viols = 0

                            eval_road_viols_batch.append(viols)

                eval_road_viols = np.hstack((eval_road_viols, eval_road_viols_batch))
                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False)

                check_result_total.append(check_result)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

                # Plot predicted timestep for current scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timesteps[0]}")
                log_writer.add_figure('eval/prediction_full', fig, i)
        check_result_total = np.concatenate(check_result_total, axis=1)
        print('violation rate {}'.format(check_result_total.sum() / check_result_total.size))
        print('RB vio {}'.format(eval_road_viols.sum() / (eval_road_viols.size * num_sample)))
        print('ade {}'.format(np.mean(eval_ade_batch_errors)))
        print('fde {}'.format(np.mean(eval_fde_batch_errors)))
        print('kde {}'.format(np.mean(eval_kde_nll)))
