import json
import os.path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pylab as plt
import requests


def fetch_avg_weight_per_rep(api_key):
    """
    Fetch data from Hevy API for a specified exercise and metric.

    Parameters:
        api_key (str): Your Hevy API key.
    """
    today = datetime.now().date()
    cache_file = f"hevy_data_{today}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data_dict = json.load(f)
        # convert timestamps back to datetime
        data_dict = {
            k: [(datetime.fromtimestamp(dt), v) for dt, v in v_list] for k, v_list in data_dict.items()
        }
        return data_dict
    url = 'https://api.hevyapp.com/v1/workouts'
    headers = {
        'accept': 'application/json',
        'api-key': api_key
    }

    params = {
        'page': 1,
        'pageSize': 10
    }

    data_dict: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
        list)  # exercise_name -> List[(timestamp, metric_value)]

    while True:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if 'error' in data and data['error'] == 'Page not found':
            # Done
            break

        if 'workouts' not in data:
            print("No workout data found.")
            break

        workouts = data['workouts']

        if not workouts:
            break

        for workout in workouts:
            title = workout['title']
            start_dt = datetime.fromisoformat(workout['start_time'].replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(workout['end_time'].replace('Z', '+00:00'))
            duration = end_dt - start_dt

            print(f"Processing {title} at {start_dt} ({duration})")

            for exercise in workout['exercises']:
                exercise_name = exercise['title']
                num_sets = len(exercise['sets'])
                if num_sets > 0:
                    weight_kg = [set_info.get('weight_kg', 0) for set_info in exercise['sets']]
                    reps = [set_info.get('reps', 0) for set_info in exercise['sets']]
                    if not any(w is None for w in weight_kg) and not any(r is None for r in reps):
                        print(f"  {exercise_name} ({num_sets} sets)")
                        median_weight = np.percentile(weight_kg, 50)
                        mask = [w >= median_weight for w in weight_kg]
                        set_volume = sum([m * w * r for m, w, r in zip(mask, weight_kg, reps)])
                        total_reps = sum([m * r for m, r in zip(mask, reps)])
                        if total_reps > 0:
                            avg_weight_kg = set_volume / total_reps
                            data_dict[exercise_name].append((start_dt, avg_weight_kg))

        params['page'] += 1

    for key in data_dict:
        data_dict[key].sort(key=lambda x: x[0])

    with open(cache_file, 'w') as f:
        # convert datetimes to unic timestamp
        json.dump(
            {
                k: [(int(dt.timestamp()), v) for dt, v in v_list] for k, v_list in data_dict.items()
            }, f, indent=2
        )

    return data_dict


if __name__ == '__main__':
    API_TOKEN = ...
    data_dict = fetch_avg_weight_per_rep(API_TOKEN)
    for key in [
        'Bench Press (Barbell)',
        'Deadlift (Barbell)',
        'Bicep Curl (Dumbbell)',
        'Shoulder Press (Dumbbell)'
    ]:
        data = data_dict.get(key, None)
        if data is None:
            print(f"No data found for {key}")
            continue
        timestamps, metrics = zip(*data)
        metrics = np.asarray(metrics)
        metrics /= np.mean(metrics)
        metrics *= 100.
        plt.plot(timestamps, metrics, lw=1, label=key)
    plt.xlabel('Timestamp')
    plt.ylabel('Avg. weight per rep (% relative to long-term mean)')
    plt.legend()
    plt.show()
