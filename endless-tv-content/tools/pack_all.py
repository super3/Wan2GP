#!/usr/bin/env python3
"""pack_all.py: every short's key frames (plus Riverside's first frame) bin-packed into as few runs as fit (pack-v2-<n>, each JSON
under 950 kB, JPEG q93 4:4:4), and the box commands from a plan.  Writes bake/v2/pack_<n>.json and bake/v2/cmd_<id>.json, prints counts.
The boxes only see the latest 50 runs, so everything for every box has to fit in one go: packs first, commands last."""
import base64, io, json, os
from PIL import Image
O = '/home/user/Wan2GP/outputs/shorts'
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bake', 'v2')
os.makedirs(B, exist_ok=True)
SLUGS = ('geometric_lab', 'retro_exaggerated', 'bold_shapes', 'desert_chase', 'wobble_suburb')
SEEDS = {'geometric_lab': 9201, 'bold_shapes': 9301, 'desert_chase': 9401, 'retro_exaggerated': 9501, 'wobble_suburb': 9601}
keys, jobs = {}, {}
for slug in SLUGS:
    for k, shot in enumerate(json.load(open(f'{O}/{slug}/shots.json'))['shots']):
        names = []
        for w in ('start', 'end'):
            rel = shot.get(w) or shot['start']
            key = f'{slug}/{os.path.splitext(rel)[0]}'
            if key not in keys:
                im = Image.open(f'{O}/{slug}/{rel}').convert('RGB'); assert im.size == (1280, 704), (rel, im.size)
                b = io.BytesIO(); im.save(b, 'JPEG', quality=93, subsampling=0); keys[key] = base64.b64encode(b.getvalue()).decode()
            names.append(key)
        jobs[f"{slug}__{shot['id']}"] = {'name': f"{slug}__{shot['id']}", 'start': names[0], 'end': names[1], 'frames': int(shot['frames']),
                                          'seed': SEEDS[slug] + k, 'prompt': shot['prompt']}
card = json.load(open(os.path.join(B, '..', 'img_cardigan2.json')))
keys['cardigan/still'] = card['jpg_b64']
for j in json.load(open(os.path.join(B, '..', 'set3_2_card.json')))['jobs'] + json.load(open(os.path.join(B, '..', 'set4_2_card.json')))['jobs']:
    jobs[j['name']] = {'name': j['name'], 'start': 'cardigan/still', 'end': 'cardigan/still', 'frames': int(j['frames']), 'seed': int(j['seed']), 'prompt': j['prompt']}
riv = json.load(open(os.path.join(B, '..', 'cmd_shA_1.json')))['jobs'][0]; assert riv['name'].startswith('riverside')
jobs[riv['name']] = riv
im = Image.open(os.path.join(B, '..', 'still_news_1280x704.png')).convert('RGB'); b = io.BytesIO(); im.save(b, 'JPEG', quality=95, subsampling=0)
keys['riverside/first_frame'] = base64.b64encode(b.getvalue()).decode()
# bin-pack, largest first
bins = []
for key, data in sorted(keys.items(), key=lambda kv: -len(kv[1])):
    for bn in bins:
        if bn['size'] + len(data) + len(key) + 40 < 950_000:
            bn['images'][key] = data; bn['size'] += len(data) + len(key) + 40; break
    else:
        bins.append({'images': {key: data}, 'size': len(data) + len(key) + 200})
for n, bn in enumerate(bins, 1):
    body = json.dumps({'id': f'pack-v2-{n}', 'kind': 'key-frames', 'images': bn['images']})
    assert len(body) < 1_000_000, len(body)
    open(os.path.join(B, f'pack_{n:02d}.json'), 'w').write(body)
# plan: round 1 one short per box, round 2 desert then wobble, balanced
plan = json.load(open(os.path.join(B, 'plan.json')))
for cid, names in plan.items():
    body = {'id': cid, 'cmd': 'jobs', 'jobs': [jobs[n] for n in names]}
    open(os.path.join(B, f'cmd_{cid}.json'), 'w').write(json.dumps(body))
    print(cid, len(names), sum(jobs[n]['frames'] for n in names))
print(len(keys), 'keys in', len(bins), 'packs;', len(plan), 'commands;', 'total runs', len(bins) + len(plan))
missing = set(jobs) - {n for v in plan.values() for n in v}; print('unplanned jobs:', sorted(missing))
