from java.awt import Color

from src.wafer.annotation import Annotation


class Focus(Annotation):
    string = "focus"
    color = Color.green
    handle_size_global = 15
    handle_size_local = 15

    def __init__(self, id_, poly):
        super(Focus, self).__init__(poly)
        self.id_ = id_

