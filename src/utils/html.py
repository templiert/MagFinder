def tag(text, tag_name):
    return "<{}>{}</{}>".format(tag_name, text, tag_name)


def print_list(*elements):
    items = "".join("<li>{}</li>".format(s) for s in elements)
    return "<ul>{}</ul>".format(items)
