from java.awt import Color

from src.wafer.annotation import Annotation


class Section(Annotation):
    string = "section"
    color = Color.blue
    handle_size_global = 15
    handle_size_local = 15

    def __init__(self, id_, poly):
        super(Section, self).__init__(poly)
        self.id_ = id_
        self.rois = {}
        self.focus = None
        self.magnet = None
        self.transform = None
        self.poly_transform = None
        self.poly_transform_inverse = None

    def all_annotations(self):
        """Returns all annotations owned by this section, including the section itself."""
        annotations = [self, self.focus, self.magnet] + [
            roi for _, roi in sorted(self.rois.iteritems())
        ]
        return filter(None, annotations)
