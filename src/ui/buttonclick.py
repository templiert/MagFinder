from java.awt.event import ActionListener


class ButtonClick(ActionListener):
    def __init__(self, field, value):
        self.field = field
        self.value = value

    def actionPerformed(self, event):
        self.field.setText(self.value)
