# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import flatbuffers

class FBFrameCameraPose(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFBFrameCameraPose(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FBFrameCameraPose()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FBFrameCameraPoseBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x63\x61\x70\x6F", size_prefixed=size_prefixed)

    # FBFrameCameraPose
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FBFrameCameraPose
    def Frame(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FBFrameCameraPose
    def Position(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .FBVector3 import FBVector3
            obj = FBVector3()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FBFrameCameraPose
    def Quaternion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .FBQuaternion import FBQuaternion
            obj = FBQuaternion()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def FBFrameCameraPoseStart(builder): builder.StartObject(3)
def FBFrameCameraPoseAddFrame(builder, frame): builder.PrependInt32Slot(0, frame, 0)
def FBFrameCameraPoseAddPosition(builder, position): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(position), 0)
def FBFrameCameraPoseAddQuaternion(builder, quaternion): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(quaternion), 0)
def FBFrameCameraPoseEnd(builder): return builder.EndObject()