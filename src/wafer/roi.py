from java.awt import Color

from src.wafer.annotation import Annotation


class ROI(Annotation):
    string = "roi"
    color = Color.yellow
    handle_size_global = 15
    handle_size_local = 2

    def __init__(self, id_section, id_roi, poly):
        super(ROI, self).__init__(poly)
        self.id_section = id_section
        self.id_roi = id_roi

    @classmethod
    def make_name(cls, id_section, id_roi):
        return "{}-{:04}.{:02}".format(cls.string, id_section, id_roi)

    @classmethod
    def make_header(cls, id_section, id_roi):
        return "{}.{:04}.{:02}".format(cls.string, id_section, id_roi)

    def __str__(self):
        return self.make_name(self.id_section, self.id_roi)

    def __repr__(self):
        return "ROI | section {} roi {} | {:.3f} {:.3f}".format(
            self.id_section, self.id_roi, self.centroid[0], self.centroid[1]
        )

    @property
    def header(self):
        return self.make_header(self.id_section, self.id_roi)
