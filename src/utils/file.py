import os

from src.utils.log import dlog

from src.constant import IMAGE_FORMATS


def mkdir_p(path):
    path = os.path.join(path, "")
    try:
        os.mkdir(path)
        dlog("Folder created: {}".format(path))
    except Exception as e:
        if e[0] == 20047:
            pass
        else:
            dlog("Exception during folder creation: {}".format(e))
    return path


def folder_content(folder):
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))]


def get_image_paths(root):
    """Returns all valid image paths in root, sorted by file size (smallest first).

    Excludes overview and annotated images.
    """
    excluded = "overview", "annotated"
    names = [
        name
        for name in os.listdir(root)
        if not any(part in name.lower() for part in excluded)
        and any(name.endswith(fmt) for fmt in IMAGE_FORMATS)
    ]
    return sorted([os.path.join(root, name) for name in names], key=os.path.getsize)


def roi_image_filename(id_):
    """Filename for an extracted high-res ROI image (magreorderer output)."""
    return "roi_{:04}.tif".format(id_)
