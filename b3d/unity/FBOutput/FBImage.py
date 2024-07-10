# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import flatbuffers

class FBImage(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFBImage(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FBImage()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FBImageBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x69\x6D\x61\x67", size_prefixed=size_prefixed)

    # FBImage
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FBImage
    def Image(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # FBImage
    def ImageAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # FBImage
    def ImageLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def FBImageStart(builder): builder.StartObject(1)
def FBImageAddImage(builder, image): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(image), 0)
def FBImageStartImageVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def FBImageEnd(builder): return builder.EndObject()
