import csv
import json 
from pathlib import Path
import os
from unittest import expectedFailure

def process_data(data_path, img_root):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    curimg=data["source_data_info"]["source_data_name_jpg"]
    curimgsz=data["source_data_info"]["document_resolution"]
    processed_data=[]
    for row_data in data["learning_data_info"]["annotation"]:
        if "visual_instruction" in row_data and "bounding_box" in row_data:
            processed_data.append([
                os.path.join(img_root,curimg), row_data["visual_instruction"], *row_data["bounding_box"],
                *curimgsz])
    return processed_data

def process_dir(dir_path,img_path,csv_path):
    dir_path = Path(dir_path)
    processed_data=[]
    for file in dir_path.glob("*.json"):
        try:
            processed_data.extend(process_data(file, img_path))
        except:
            pass
    with open(csv_path, "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "visual_instruction", "x", "y", "w", "h","img_w","img_h"])
        writer.writerows(processed_data)

    return processed_data



if __name__ == "__main__":
    from itertools import product
    for p1,p2 in product(["valid","train"],["press","report"]):
        process_dir(f"data/{p1}/{p2}_json",f"data/{p1}/{p2}_jpg",f"{p1}_{p2}.csv")
    
    #Merge all train and valid
    # Merge all train CSVs into train_all.csv and all valid CSVs into valid_all.csv
    def merge_csvs(files, out_path):
        all_rows = []
        for f in files:
            with open(f, "r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                header = next(reader)
                for row in reader:
                    all_rows.append(row)
        with open(out_path, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(all_rows)

    train_csvs = [f"train_press.csv", f"train_report.csv"]
    valid_csvs = [f"valid_press.csv", f"valid_report.csv"]
    merge_csvs(train_csvs, "train_all.csv")
    merge_csvs(valid_csvs, "valid_all.csv")
    # print(data)
