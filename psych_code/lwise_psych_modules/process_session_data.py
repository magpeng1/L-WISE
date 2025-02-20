import json
import xarray as xr
import collections
import numpy as np
import argparse
from urllib.parse import unquote
import pprint
import re


def process_nafc_assignment_json(asn_json: dict, verbose=False):
    """
    Asn_json
    :param asn_json:
    :return:
    """
    answer_text = asn_json['Answer']
    answer_text = answer_text.split('<FreeText>')[1].split('</FreeText>')[0]

    decoded_text = unquote(answer_text)

    data = (json.loads(decoded_text))

    bonus = get_bonus_amount_from_message(data)

    ds = process_session_data(
        session_data=data['trials'],
    )

    if verbose or ds is None:
        pprint.PrettyPrinter(indent=2).pprint(data)

    return ds, bonus

def compute_bonus_amount(ds:xr.Dataset):
    ngt = ds.perf.sum().item()
    bonus_usd_earned = (ds.bonus_usd_if_correct * ds.perf).sum().item()
    print(f'${bonus_usd_earned} USD bonus earned ({int(ngt)} correct)')
    return bonus_usd_earned

def get_bonus_amount_from_message(json_data):
    try:
        message = json_data['trials'][-2]['stimulus']
    except KeyError:
        message = json_data['trials'][-3]['stimulus']
    pattern = r'\$\d+(?:\.\d{2})?'
    matches = re.findall(pattern, message)

    if not matches:
        return None
    elif len(matches) > 1:
        raise ValueError("More than one dollar amount mentioned in the message shown to the participant. Yes, I agree that this is not an ideal way to implement bonuses.")

    return float(matches[0][1:])

def process_session_data(session_data:list):

    data_vars = collections.defaultdict(list)
    coords = collections.defaultdict(list)


    data_var_names = [
        'perf',
        'i_choice',
        'reaction_time_msec',
    ]
    coord_var_names = [
        'i_correct_choice',
        'bonus_usd_if_correct',
        'timestamp_start',
        'rel_timestamp_response',
        'choice_duration_msec',
        'stimulus_image_url',
        'stimulus_image_url_l',
        'stimulus_image_url_r',
        'class',
        'class_l',
        'class_r',
        'choice_image_urls',
        'stimulus_name',
        'choice_name',
        'query_string',
        'stimulus_duration_msec',
        'post_stimulus_delay_duration_msec',
        'pre_choice_lockout_delay_duration_msec',
        'keep_stimulus_on',
        'stimulus_width_px',
        'choice_width_px',
        'monitor_width_px',
        'monitor_height_px',
        'mask_image_url',
        'mask_duration_msec',
        'block',
        'condition_idx',
        'trial_type',
    ]
    data_var_names = sorted(set(data_var_names))
    coord_var_names = sorted(set(coord_var_names))

    for jspsych_trial in session_data:
        if 'trial_outcome' not in jspsych_trial:
            continue

        dat = jspsych_trial['trial_outcome']

        for name in data_var_names:
            if name in dat:
                data_vars[name].append(dat[name])
            else:
                data_vars[name].append(np.nan)

        for name in coord_var_names:
            if name in dat:
                coords[name].append(dat[name])
            else:
                coords[name].append(np.nan)

    for k in data_vars:
        data_vars[k] = (['obs'], data_vars[k])
    for k in coords:
        if k == 'choice_image_urls':
            max_nchoices = max([len(x) for x in coords[k]])
            # Pad with '' to make all lists the same length
            for i in range(len(coords[k])):
                coords[k][i] += [''] * (max_nchoices - len(coords[k][i]))
            coords[k] = (['obs', 'choice_slot'], coords[k])
        else:
            coords[k] = (['obs'], coords[k])


    if len(data_vars) == 0:
        return None
    ds = xr.Dataset(
        data_vars = data_vars,
        coords = coords,
    )

    pvalues = [float(p) if p is not None else np.nan for p in ds['perf'].values]
    ds['perf'].values[:] = pvalues

    ds['obs'] = np.arange(len(ds['obs']))
    return ds


def main():
  parser = argparse.ArgumentParser(description='Convert data to xarray format from JSON')
  parser.add_argument('--file_path', type=str, required=True, help='Path of json file')
  args = parser.parse_args()


  json_data = json.load(open(args.file_path, 'r'))

  ds = process_session_data(json_data)

  print(ds)

  # bonus = compute_bonus_amount(ds)
  bonus = get_bonus_amount_from_message(json_data)
  print("Bonus earned:$" + str(bonus))


if __name__ == "__main__":
  main()

