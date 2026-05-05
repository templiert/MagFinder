from java.awt import Color

from src.wafer.annotation import Annotation


class Magnet(Annotation):
    string = "magnet"
    color = Color.green
    handle_size_global = 15
    handle_size_local = 15

    def __init__(self, id_, poly):
        super(Magnet, self).__init__(poly)
        self.id_ = id_

