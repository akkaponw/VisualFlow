from pascal_voc_writer import Writer
from polygon_pascalvoc_writer import VocWriter
from tqdm import tqdm
import time
import os
from PIL import Image
import xml.etree.ElementTree as ET
import json
import numpy as np

# Convert Yolo bb to Pascal_voc bb
def yolo2pascalvoc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    xmin = ((2 * x_center * image_w) - w)/2
    ymin = ((2 * y_center * image_h) - h)/2
    xmax = xmin + w
    ymax = ymin + h
    return [xmin, ymin, xmax, ymax]

def coco2pascalvoc(xmin, ymin, w, h):
    return [xmin,ymin, xmin + w, ymin + h]


def pascalvoc2yolo(xmin, ymin, xmax, ymax, image_w, image_h):
    x_center = ((xmax + xmin)/(2*image_w))
    y_center = ((ymax + ymin)/(2*image_h))
    w = (xmax - xmin)/image_w
    h = (ymax - ymin)/image_h
    return [x_center, y_center, w, h]

def yolo2coco(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    xmin = ((2 * x_center * image_w) - w)/2
    ymin = ((2 * y_center * image_h) - h)/2
    return [xmin, ymin, w, h]

def pascalvoc2coco(xmin, ymin, xmax, ymax):
    w = xmax-xmin
    h = ymax - ymin
    return [xmin,ymin, w, h]

def coco2yolo(xmin, ymin, w, h, image_w, image_h):
    x_center = ((2*xmin + w)/(2*image_w))
    y_center = ((2*ymin + h)/(2*image_h))
    return [x_center , y_center, w/image_w, h/image_h]


# image_path = os.path.join(image_dir, image_name)
#             image_name, image_ext = os.path.splitext(image_name)
#             label_name, label_ext = image_name, ".txt"
#             label_path = os.path.join(labels_dir, label_name + label_ext)
#             new_image_name = f"{image_name}_rotate_{angle}{image_ext}"
#             new_label_name = f"{label_name}_rotate_{angle}{label_ext}"
#             new_image_path = os.path.join(output_images_dir, new_image_name)
#             new_label_path = os.path.join(output_labels_dir, new_label_name)
#             image = Image.open(image_path)
#             image = image.convert("RGBA")
#             image_width, image_height = image.size
#             new_image = image.copy()
#             draw = ImageDraw.Draw(new_image)
#             bboxes = read_yolo_txt_file(label_path)
#             converted_bboxes = []
#             new_bboxes = []
#             for bbox in bboxes:
#                 radian_angle = math.radians(angle)
#                 cos_theta = math.cos(radian_angle)
#                 sin_theta = math.sin(radian_angle)
#                 new_xmin = int(cx + (xmin - cx) * c
#                 unnormalized_bbox = [bbox[0], bbox[1]*image_width, bbox[2]*image_height, bbox[3]*image_width, bbox[4]*image_height]
#                 new_width = unnormalized_bbox[3]*cos_theta + unnormalized_bbox[4]*sin_theta
#                 new_height = unnormalized_bbox[4]*cos_theta + unnormalized_bbox[3]*sin_theta
#                 new_xcenter = new_width / 2
#                 new_ycenter = new_height / 2
#                 new_bbox = [bbox[0], new_xcenter / image_width, new_ycenter / image_height, new_width / image_width, new_height / image_height]

def to_voc(in_format=None, images=None, annotations=None, class_file=None, out_dir=None, json_file=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('yolo' or 'coco')")
    if out_dir is None:
        raise ValueError("Missing argument: out_dir")
    if images is None:
        raise ValueError("Missing argument: images")
    if not os.path.isdir(os.path.join(out_dir, "xmls")):
        os.mkdir(os.path.join(out_dir, "xmls"))
    if in_format == "yolo":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt")
        if annotations is None:
            raise ValueError("Missing argument: annotations")
        else:
            class_dict = {}
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    class_dict[line_num] = name

            image_info = []
            image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            with tqdm(total=len(image_files), desc="Converting...") as pbar:
                for filename in os.listdir(images):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(images, filename)
                        label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".txt")
                        try:
                            image_width, image_height = 0, 0
                            with Image.open(image_path) as img:
                                image_width, image_height = img.size
                            with open(label_path, 'r') as label:
                                bboxes = []
                                for line in label:
                                    line = line.strip()
                                    numbers = line.split()
                                    class_name = class_dict[int(numbers[0])]
                                    x_center = float(numbers[1])
                                    y_center = float(numbers[2])
                                    w = float(numbers[3])
                                    h = float(numbers[4])
                                    bboxes.append([x_center, y_center, w, h, class_name])
                                writer = Writer(image_path, image_width, image_height)
                                for bbox in bboxes:
                                    voc_bbox = yolo2pascalvoc(bbox[0], bbox[1], bbox[2], bbox[3], image_width, image_height)
                                    writer.addObject(bbox[4], voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[3])
                                writer.save(os.path.join(output, "xmls", filename.rsplit(".", 1)[0] + ".xml"))
                        except OSError:
                            print(image_path + " skipped")
                            pass
                        pbar.update(1)
            print("Completed!")

    if in_format == "coco":
        if json_file is None:
            raise ValueError("Missing argument: json_file. Please provide path to json file")
        else:
            print("starting")
            bbox_mapping = {}
            class_mapping = {}
            coco = json.load(open(json_file, 'r'))
            image_ids = []
            for c in coco['categories']:
                class_mapping[c["id"] - 1] = c["name"]
            for ann in coco['annotations']:
                image_id = ann['image_id']
                if image_id not in bbox_mapping:
                    bbox_mapping[image_id] = []
                xmin = ann['bbox'][0]
                ymin = ann['bbox'][1]
                w = ann['bbox'][2]
                h = ann['bbox'][3]
                seg = []
                isseg = 0
                if ann['segmentation'] != []:
                    seg = ann['segmentation']
                    seg = np.reshape(seg, (-1, 2))
                    isseg = 1
                bbox_mapping[image_id].append([ann["category_id"] - 1, xmin, ymin, w, h, isseg, seg])
            with tqdm(total=len(coco['images']), desc="Converting...") as pbar:
                for im in coco['images']:
                    result = []
                    bboxes = bbox_mapping[im["id"]]
                    image_path = os.path.join(images, os.path.basename(im["file_name"]))
                    writer = VocWriter(images, annotations, im["file_name"])
                    for bbox in bboxes:
                        voc_bbox = coco2pascalvoc(bbox[1], bbox[2], bbox[3], bbox[4])
                        if bbox[5] == 1:
                            writer.addPolygon(class_mapping[bbox[0]], bbox[6])
                        else:
                            writer.addBndBox(class_mapping[bbox[0]], voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[3])
                    xml_filename = os.path.basename(im["file_name"]).rsplit(".", 1)[0]
                    writer.save()
                    pbar.update(1)
            print("Completed!")


def to_yolo(in_format=None, images=None, annotations=None, out_dir=None, json_file=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('voc' or 'coco')")
    if out_dir is None:
        raise ValueError("Missing argument: out_dir")
    if images is None:
        raise ValueError("Missing argument: images")
    if not os.path.isdir(os.path.join(out_dir, "labels")):
        os.mkdir(os.path.join(out_dir, "labels"))
    if in_format == "voc":
        if annotations is None:
            raise ValueError("Missing argument: annotations")
        else:
            class_lst = []
            class_path = os.path.join(out_dir, "classes.txt")
            image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            with tqdm(total=len(image_files), desc="Converting...") as pbar:
                for filename in os.listdir(images):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(images, filename)
                        label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".xml")
                        try:
                            image_width, image_height = 0, 0
                            with Image.open(image_path) as img:
                                image_width, image_height = img.size
                            result = []
                            tree = ET.parse(label_path)
                            root = tree.getroot()
                            height = int(root.find("size").find("height").text)
                            width = int(root.find("size").find("width").text)
                            for obj in root.findall('object'):
                                label = obj.find("name").text
                                if label not in class_lst:
                                    class_lst.append(label)
                                index = class_lst.index(label)
                                bbox = [float(x.text) for x in obj.find("bndbox")]
                                yolo_bbox = pascalvoc2yolo(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
                                bbox_string = " ".join([str(x) for x in yolo_bbox])
                                result.append(f"{index} {bbox_string}")

                            if result:
                                label_name = filename.rsplit(".", 1)[0] + ".txt"
                                with open(os.path.join(out_dir, "labels", label_name), "w", encoding="utf-8") as f:
                                    f.write("\n".join(result))
                        except OSError:
                            print(image_path + " skipped")
                            pass
                        pbar.update(1)
                with open(class_path, 'w', encoding='utf8') as f:
                    for item in class_lst:
                        f.write(item + '\n')
            print("Completed!")

    if in_format == "coco":
        if json_file is None:
            raise ValueError("Missing argument: json_file. Please provide path to json file")
        else:
            print("starting")
            bbox_mapping = {}
            class_mapping = {}
            coco = json.load(open(json_file, 'r'))
            image_ids = []
            for c in coco['categories']:
                class_mapping[c["id"] - 1] = c["name"]
            for ann in coco['annotations']:
                image_id = ann['image_id']
                if image_id not in bbox_mapping:
                    bbox_mapping[image_id] = []
                xmin = ann['bbox'][0]
                ymin = ann['bbox'][1]
                w = ann['bbox'][2]
                h = ann['bbox'][3]
                bbox_mapping[image_id].append([ann["category_id"] - 1, xmin, ymin, w, h])
            with tqdm(total=len(coco['images']), desc="Converting...") as pbar:
                for im in coco['images']:
                    result = []
                    bboxes = bbox_mapping[im["id"]]
                    for bbox in bboxes:
                        yolo_bbox = coco2yolo(bbox[1], bbox[2], bbox[3], bbox[4], im["width"], im["height"])
                        bbox_string = " ".join([str(x) for x in yolo_bbox])
                        result.append(f"{bbox[0]} {bbox_string}")
                    if result:
                        image_filename = os.path.basename(im["file_name"]).rsplit(".", 1)[0]

                        with open(os.path.join(out_dir, "labels", f"{image_filename}.txt"), "w", encoding="utf-8") as f:
                            f.write("\n".join(result))
                    pbar.update(1)
            print("Completed!")



def to_coco(in_format=None, images=None, annotations=None, class_file=None, output_file_path=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('yolo' or 'voc')")
    if output_file_path is None:
        raise ValueError("Missing argument: output_file_path")
    if images is None:
        raise ValueError("Missing argument: images")
    if annotations is None:
        raise ValueError("Missing argument: annotations")
    if in_format == "yolo":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt which has a list of all your classes (one class on each line)")
        coco = {"images": [{}], "categories": [], "annotations": [{}]}
        image_id = 0
        ann_id = 1
        image_info = []
        ann_info = []
        image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        with tqdm(total=len(image_files), desc="Converting...") as pbar:
            for filename in os.listdir(images):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(images, filename)
                    label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".txt")
                    try:
                        image_width, image_height = 0, 0
                        with Image.open(image_path) as img:
                            image_width, image_height = img.size
                            image_stats = {
                                            "file_name": image_path,
                                            "height": image_height,
                                            "width": image_width,
                                            "id": image_id,
                                        }
                            image_info.append(image_stats)
                        with open(label_path, 'r') as label:
                            bboxes = []
                            for line in label:
                                line = line.strip()
                                numbers = line.split()
                                class_id = int(numbers[0]) + 1
                                x_center = float(numbers[1])
                                y_center = float(numbers[2])
                                w = float(numbers[3])
                                h = float(numbers[4])
                                coco_bbox = yolo2coco(x_center, y_center, w, h,  image_width, image_height)
                                area = coco_bbox[2]*coco_bbox[3]
                                bbox_stats = {
                                            "id": ann_id,
                                            "image_id": image_id,
                                            "bbox": (coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]),
                                            "area": area,
                                            "iscrowd": 0,
                                            "category_id": class_id,
                                        }
                                ann_info.append(bbox_stats)
                                ann_id = ann_id + 1
                        image_id = image_id + 1
                    except OSError:
                        print(image_path + " skipped")
                        pass
                    pbar.update(1)
            coco["images"] = image_info
            coco["annotations"] = ann_info
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    categories = {
                        "supercategory": "Null",
                        "id": line_num + 1,
                        "name": name,
                    }
                    coco["categories"].append(categories)
            with open(output_file_path, "w") as outfile:
                json.dump(coco, outfile, indent=4)
        print("Completed!")
    
    if in_format == "voc":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt which has a list of all your classes (one class on each line)")
        coco = {"images": [{}], "categories": [], "annotations": [{}]}
        image_id = 0
        ann_id = 1
        image_info = []
        ann_info = []
        class_dict = {}
        with open(class_file, 'r') as file:
            for line_num, name in enumerate(file):
                name = name.strip()
                class_dict[name] = line_num + 1
        image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        with tqdm(total=len(image_files), desc="Converting...") as pbar:
            for filename in os.listdir(images):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(images, filename)
                    label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".xml")
                    tree = ET.parse(label_path)
                    root = tree.getroot()
                    image_height = int(root.find("size")[0].text)
                    image_width = int(root.find("size")[1].text)
                    image_channels = int(root.find("size")[2].text)
                    try:
                        image_width, image_height = 0, 0
                        with Image.open(image_path) as img:
                            image_width, image_height = img.size
                            image_stats = {
                                            "file_name": image_path,
                                            "height": image_height,
                                            "width": image_width,
                                            "id": image_id,
                                        }
                            image_info.append(image_stats)
                        for member in root.findall('object'):
                            class_name = member[0].text
                            xmin = float(member[4][0].text)
                            ymin = float(member[4][1].text)
                            xmax = float(member[4][2].text)
                            ymax = float(member[4][3].text)
                            coco_bbox = pascalvoc2coco(xmin,ymin,xmax,ymax)
                            area = coco_bbox[2]*coco_bbox[3]
                            bbox_stats = {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "bbox": (coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]),
                                        "area": area,
                                        "iscrowd": 0,
                                        "category_id": class_dict[class_name],
                                    }
                            ann_info.append(bbox_stats)
                            ann_id = ann_id + 1
                        image_id = image_id + 1
                    except OSError:
                        print(image_path + " skipped")
                        pass
                    pbar.update(1)
            coco["images"] = image_info
            coco["annotations"] = ann_info
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    categories = {
                        "supercategory": "Null",
                        "id": line_num + 1,
                        "name": name,
                    }
                    coco["categories"].append(categories)
            with open(output_file_path, "w") as outfile:
                json.dump(coco, outfile, indent=4)
        print("Completed!")
