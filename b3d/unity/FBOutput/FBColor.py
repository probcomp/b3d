# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import flatbuffers

class FBColor(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFBColor(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FBColor()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FBColorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x63\x64\x69\x63", size_prefixed=size_prefixed)

    # FBColor
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FBColor
    def R(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # FBColor
    def G(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # FBColor
    def B(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # FBColor
    def A(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def FBColorStart(builder): builder.StartObject(4)
def FBColorAddR(builder, r): builder.PrependFloat32Slot(0, r, 0.0)
def FBColorAddG(builder, g): builder.PrependFloat32Slot(1, g, 0.0)
def FBColorAddB(builder, b): builder.PrependFloat32Slot(2, b, 0.0)
def FBColorAddA(builder, a): builder.PrependFloat32Slot(3, a, 0.0)
def FBColorEnd(builder): return builder.EndObject()
