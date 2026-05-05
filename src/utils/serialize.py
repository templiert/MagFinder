from java.io import (
    FileInputStream,
    FileOutputStream,
    ObjectInputStream,
    ObjectOutputStream,
)


def serialize(x, path):
    stream = ObjectOutputStream(FileOutputStream(path))
    try:
        stream.writeObject(x)
    finally:
        stream.close()


def serialize_parallel(atom, objects, paths):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k < len(paths):
            serialize(objects[k], paths[k])


def deserialize(path):
    stream = ObjectInputStream(FileInputStream(path))
    try:
        return stream.readObject()
    finally:
        stream.close()


def deserialize_parallel(atom, paths, objects):
    while atom.get() < len(paths):
        k = atom.getAndIncrement()
        if k < len(paths):
            objects[k] = deserialize(paths[k])
