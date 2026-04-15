"""
Download D4RL MuJoCo datasets without requiring the d4rl package.
Downloads the HDF5 datasets directly from the D4RL servers and converts
them to the pickle format expected by the Decision Transformer code.
"""

import os
import collections
import pickle
import urllib.request
import requests

import numpy as np
import h5py


DATASET_URLS = {
    'hopper-medium-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5',
    'halfcheetah-medium-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5',
    'walker2d-medium-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5',
    'hopper-medium-replay-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5',
    'halfcheetah-medium-replay-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5',
    'walker2d-medium-replay-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5',
    'hopper-expert-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5',
    'halfcheetah-expert-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5',
    'walker2d-expert-v2': 'http://128.32.244.190/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5',
}


def download_dataset(name, url, data_dir='.'):
    """Download an HDF5 dataset and convert to pickle format."""
    hdf5_path = os.path.join(data_dir, f'{name}.hdf5')
    pkl_path = os.path.join(data_dir, f'{name}.pkl')
    
    if os.path.exists(pkl_path):
        print(f'  {name}.pkl already exists, skipping.')
        return

    print(f'  Downloading {name}...')
    try:
        from tqdm import tqdm
        response = requests.get(url, headers={'Host': 'rail.eecs.berkeley.edu'}, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(hdf5_path, 'wb') as file, tqdm(
            desc=name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
    except Exception:
        response = requests.get(url, headers={'Host': 'rail.eecs.berkeley.edu'})
        response.raise_for_status()
        with open(hdf5_path, 'wb') as file:
            file.write(response.content)
    print(f'  Downloaded to {hdf5_path}')

    # Convert to pickle
    with h5py.File(hdf5_path, 'r') as f:
        dataset = {k: f[k][:] for k in f.keys() if isinstance(f[k], h5py.Dataset)}

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    
    use_timeouts = 'timeouts' in dataset
    
    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000 - 1)
        
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            if k in dataset:
                data_[k].append(dataset[k][i])
        
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {k: np.array(v) for k, v in data_.items()}
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'  Samples: {num_samples}, Trajectories: {len(paths)}')
    print(f'  Returns: mean={np.mean(returns):.1f}, std={np.std(returns):.1f}, max={np.max(returns):.1f}, min={np.min(returns):.1f}')

    with open(pkl_path, 'wb') as f:
        pickle.dump(paths, f)
    print(f'  Saved to {pkl_path}')
    
    # Clean up HDF5
    os.remove(hdf5_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None,
                        help='Specific env to download (hopper, halfcheetah, walker2d). Downloads all if not specified.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset type (medium, medium-replay, expert). Downloads all if not specified.')
    args = parser.parse_args()

    # Filter datasets based on args
    datasets_to_download = {}
    for name, url in DATASET_URLS.items():
        env = name.split('-')[0]
        # dataset type is everything between env and v2
        dtype = '-'.join(name.split('-')[1:-1])
        
        if args.env and env != args.env:
            continue
        if args.dataset and dtype != args.dataset:
            continue
        datasets_to_download[name] = url

    if not datasets_to_download:
        print('No matching datasets found.')
    else:
        print(f'Will download {len(datasets_to_download)} dataset(s):')
        for name in datasets_to_download:
            print(f'  - {name}')
        print()

        # Ensure h5py is installed
        try:
            import h5py
        except ImportError:
            print('Installing h5py...')
            os.system('pip install h5py')
            import h5py

        data_dir = os.path.dirname(os.path.abspath(__file__))
        for name, url in datasets_to_download.items():
            download_dataset(name, url, data_dir)
        
        print('\nDone!')
