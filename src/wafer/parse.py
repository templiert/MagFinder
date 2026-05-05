from src.wafer.focus import Focus
from src.wafer.landmark import Landmark
from src.wafer.magnet import Magnet
from src.wafer.roi import ROI
from src.wafer.section import Section

_CLASSES_BY_STRING = {
    cls.string: cls for cls in (Section, ROI, Focus, Magnet, Landmark)
}


def parse_annotation_name(name, delimiter_0="-", delimiter_1="."):
    """Parses an annotation name into (class, section_id, annotation_id).

    For non-ROI types, annotation_id == section_id (unused by callers).

    Examples:
        parse_annotation_name("section-0012")        -> (Section,  12, 12)
        parse_annotation_name("roi-0019.01")          -> (ROI,      19,  1)
        parse_annotation_name("focus-0003")           -> (Focus,     3,  3)
        parse_annotation_name("section.0012", ".")    -> (Section,  12, 12)
    """
    parts = name.split(delimiter_0)
    cls = _CLASSES_BY_STRING[parts[0]]
    ids = [int(x) for x in delimiter_0.join(parts[1:]).split(delimiter_1)]
    if cls is ROI:
        return cls, ids[0], ids[1]
    return cls, ids[0], ids[0]
