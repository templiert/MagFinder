from java.awt import Color

from src.wafer.annotation import Annotation


class Landmark(Annotation):
    string = "landmark"
    color = Color.yellow
    handle_size_global = 15
    handle_size_local = 15

    def __init__(self, id_, poly):
        super(Landmark, self).__init__(poly)
        self.id_ = id_

