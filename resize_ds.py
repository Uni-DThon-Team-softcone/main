import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp

tgt="valid"
input_csv = f"{tgt}_all.csv"
output_csv = f"{tgt}_all_resized.csv"
resized_img_dir = "resized_images"
target_width = 1024

os.makedirs(resized_img_dir, exist_ok=True)

def get_target_size(orig_w, orig_h, target_width):
    if orig_w == 0 or orig_w == target_width:
        return int(orig_w), int(orig_h)
    scale = target_width / orig_w
    nh = max(8, int(round(orig_h * scale)))
    return target_width, nh

def process_row(args):
    row, fieldnames = args
    rel_img_path = row.get("image_path") or ""
    if not rel_img_path:
        return None
    orig_img_path = rel_img_path
    if not os.path.isabs(orig_img_path):
        orig_img_path = os.path.join(os.path.dirname(input_csv), rel_img_path)
    if not os.path.exists(orig_img_path):
        print(f"Skip missing image: {orig_img_path}")
        return None
    try:
        with Image.open(orig_img_path) as img:
            orig_w, orig_h = img.size
            resized_w, resized_h = get_target_size(orig_w, orig_h, target_width)
            out_img_name = os.path.basename(rel_img_path)
            out_img_path = os.path.join(resized_img_dir, out_img_name)
            # Skip if the resized image already exists
            if os.path.exists(out_img_path):
                pass  # Do not overwrite, just update row fields below
            else:
                if (orig_w, orig_h) != (resized_w, resized_h):
                    img = img.resize((resized_w, resized_h), Image.BILINEAR)
                img.save(out_img_path)
    except Exception as e:
        print(f"Failed processing {orig_img_path}: {e}")
        return None
    row_out = row.copy()
    row_out['image_path'] = os.path.join(resized_img_dir, out_img_name).replace("\\", "/")
    row_out['img_w'] = str(resized_w)
    row_out['img_h'] = str(resized_h)

    scale_x = resized_w / orig_w if orig_w else 1.0
    scale_y = resized_h / orig_h if orig_h else 1.0
    try:
        x = float(row.get('x', 0)) * scale_x
        y = float(row.get('y', 0)) * scale_y
        w = float(row.get('w', 0)) * scale_x
        h = float(row.get('h', 0)) * scale_y
        row_out['x'] = str(x)
        row_out['y'] = str(y)
        row_out['w'] = str(w)
        row_out['h'] = str(h)
    except (TypeError, ValueError):
        pass
    # Remove old img_w/img_h if in row but not in fieldnames
    if 'img_w' in row_out and 'img_w' not in fieldnames:
        del row_out['img_w']
    if 'img_h' in row_out and 'img_h' not in fieldnames:
        del row_out['img_h']
    return row_out
with open(input_csv, "r", encoding="utf-8") as f_in, open(output_csv, "w", newline='', encoding="utf-8") as f_out:
    reader = list(csv.DictReader(f_in))
    if not reader or not reader[0]:
        raise RuntimeError("No columns found in input csv")
    fieldnames = list(reader[0].keys())
    out_fields = []
    for col in fieldnames:
        if col in ('img_w', 'img_h'):
            continue
        out_fields.append(col)
    out_fields += ['img_w', 'img_h']
    writer = csv.DictWriter(f_out, fieldnames=out_fields)
    writer.writeheader()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        row_args = [(row, fieldnames) for row in reader]
        for row_out in tqdm(pool.imap_unordered(process_row, row_args), total=len(row_args)):
            if row_out is None:
                continue
            writer.writerow({k: row_out.get(k, "") for k in out_fields})
