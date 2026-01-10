'''
Helper script that converts all the Pascal VOC XML files from Roboflow into .txt files
Additionally, it will convert the data inside each XML file into a format that will be used by Qwen and the JSONL files

Pascal VOC is class-name xmin ymin xmax ymax

'''
import os
import shutil
import xml.etree.ElementTree as ET

INPUT_DIR = "/home/troy/jam-causing-material-V2/test/labels"
OUTPUT_TXT_DIR = "/home/troy/jam-causing-material-V2/test/labels"
ARCHIVE_XML_DIR = "/home/troy/jam-causing-material-V2-xml"

os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_XML_DIR, exist_ok=True)

def xml_to_txt(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")

        if name is None or bndbox is None:
            continue

        xmin = bndbox.findtext("xmin")
        ymin = bndbox.findtext("ymin")
        xmax = bndbox.findtext("xmax")
        ymax = bndbox.findtext("ymax")

        if None in (xmin, ymin, xmax, ymax): # Was incorrect previously!
            continue

        lines.append(f"{name} {xmin} {ymin} {xmax} {ymax}") # Was incorrect previously!

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".xml"):
        xml_path = os.path.join(INPUT_DIR, file)
        txt_name = os.path.splitext(file)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_TXT_DIR, txt_name)
        xml_to_txt(xml_path, txt_path)

        # Move XML after successful conversion
        dst_xml_path = os.path.join(ARCHIVE_XML_DIR, file)
        shutil.move(xml_path, dst_xml_path)

print("Conversion complete. XML files moved.")

