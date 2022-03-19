# Found this program for writing a list of image names into a csv file on Reddit
# Source: https://www.reddit.com/r/learnpython/comments/p1y31r/reading_image_file_names_from_directory_into_csv/

from pathlib import Path
import csv

img_folder = Path("c:\\Users\\bsawe\\Documents\\GitHub\\ML-GraphVertexEdgeCounter\\testdata")
img_files = [file.name for file in img_folder.rglob("*.*")]
csv_file = img_folder / "verticesCount.csv"

with open(csv_file, "w", newline="") as csv_file_object:
    csv_writer = csv.writer(csv_file_object, delimiter=',')
    for row in img_files:
        csv_writer.writerow([row])