# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FBOutput

import flatbuffers

class FBObjectCatalog(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFBObjectCatalog(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FBObjectCatalog()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FBObjectCatalogBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x63\x61\x74\x61", size_prefixed=size_prefixed)

    # FBObjectCatalog
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FBObjectCatalog
    def ObjectCatalog(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # FBObjectCatalog
    def ObjectCatalogLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def FBObjectCatalogStart(builder): builder.StartObject(1)
def FBObjectCatalogAddObjectCatalog(builder, objectCatalog): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(objectCatalog), 0)
def FBObjectCatalogStartObjectCatalogVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FBObjectCatalogEnd(builder): return builder.EndObject()