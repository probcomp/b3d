# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import flatbuffers

class FBVector3(object):
    __slots__ = ['_tab']

    # FBVector3
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FBVector3
    def X(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # FBVector3
    def Y(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))
    # FBVector3
    def Z(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8))

def CreateFBVector3(builder, x, y, z):
    builder.Prep(4, 12)
    builder.PrependFloat32(z)
    builder.PrependFloat32(y)
    builder.PrependFloat32(x)
    return builder.Offset()
