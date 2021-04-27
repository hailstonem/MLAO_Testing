# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: doptical/api/scanner.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='doptical/api/scanner.proto',
  package='scanner',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1a\x64optical/api/scanner.proto\x12\x07scanner\"\x07\n\x05\x45mpty\"\x1d\n\x0fScannerResponse\x12\n\n\x02id\x18\x01 \x01(\t\"\x1c\n\x08\x46ilename\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\"5\n\rScannerStatus\x12\x13\n\x0bstatus_code\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"F\n\x0cScannerRange\x12\x0c\n\x04xmin\x18\x01 \x01(\x02\x12\x0c\n\x04xmax\x18\x02 \x01(\x02\x12\x0c\n\x04ymin\x18\x03 \x01(\x02\x12\x0c\n\x04ymax\x18\x04 \x01(\x02\")\n\x11ScannerPixelRange\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"4\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\"\x1a\n\x0cImageStackID\x12\n\n\x02id\x18\x01 \x01(\t\"(\n\x06Images\x12\x1e\n\x06images\x18\x01 \x03(\x0b\x32\x0e.scanner.Image\"\x1e\n\x0cImagesLength\x12\x0e\n\x06length\x18\x01 \x01(\x05\"\x16\n\x05\x41rray\x12\r\n\x05\x61rray\x18\x01 \x01(\x0c\"\x15\n\x08PiezoPos\x12\t\n\x01z\x18\x01 \x01(\x02\"1\n\x0cZernikeModes\x12\r\n\x05modes\x18\x01 \x03(\x05\x12\x12\n\namplitudes\x18\x02 \x03(\x02\"\x1a\n\x07Timeout\x12\x0f\n\x07timeout\x18\x01 \x01(\x02\x32\xa1\x06\n\x07Scanner\x12\x37\n\tStartScan\x12\x0e.scanner.Empty\x1a\x18.scanner.ScannerResponse\"\x00\x12-\n\tStartLive\x12\x0e.scanner.Empty\x1a\x0e.scanner.Empty\"\x00\x12,\n\x08StopScan\x12\x0e.scanner.Empty\x1a\x0e.scanner.Empty\"\x00\x12\x33\n\x0f\x43learScanImages\x12\x0e.scanner.Empty\x1a\x0e.scanner.Empty\"\x00\x12.\n\x07\x43\x61pture\x12\x10.scanner.Timeout\x1a\x0f.scanner.Images\"\x00\x12\x37\n\x0cSetScanRange\x12\x15.scanner.ScannerRange\x1a\x0e.scanner.Empty\"\x00\x12\x41\n\x11SetScanPixelRange\x12\x1a.scanner.ScannerPixelRange\x1a\x0e.scanner.Empty\"\x00\x12\x37\n\x0cGetScanRange\x12\x0e.scanner.Empty\x1a\x15.scanner.ScannerRange\"\x00\x12\x41\n\x11GetScanPixelRange\x12\x0e.scanner.Empty\x1a\x1a.scanner.ScannerPixelRange\"\x00\x12\x39\n\rGetScanImages\x12\x15.scanner.ImageStackID\x1a\x0f.scanner.Images\"\x00\x12>\n\x13GetScanImagesLength\x12\x0e.scanner.Empty\x1a\x15.scanner.ImagesLength\"\x00\x12\x35\n\x11GetScanImageBytes\x12\x0e.scanner.Empty\x1a\x0e.scanner.Array\"\x00\x12\x32\n\x0bSetPiezoPos\x12\x11.scanner.PiezoPos\x1a\x0e.scanner.Empty\"\x00\x12=\n\x12SetSLMZernikeModes\x12\x15.scanner.ZernikeModes\x1a\x0e.scanner.Empty\"\x00\x62\x06proto3'
)




_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='scanner.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=46,
)


_SCANNERRESPONSE = _descriptor.Descriptor(
  name='ScannerResponse',
  full_name='scanner.ScannerResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='scanner.ScannerResponse.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=77,
)


_FILENAME = _descriptor.Descriptor(
  name='Filename',
  full_name='scanner.Filename',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='filename', full_name='scanner.Filename.filename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=79,
  serialized_end=107,
)


_SCANNERSTATUS = _descriptor.Descriptor(
  name='ScannerStatus',
  full_name='scanner.ScannerStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status_code', full_name='scanner.ScannerStatus.status_code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message', full_name='scanner.ScannerStatus.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=109,
  serialized_end=162,
)


_SCANNERRANGE = _descriptor.Descriptor(
  name='ScannerRange',
  full_name='scanner.ScannerRange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='xmin', full_name='scanner.ScannerRange.xmin', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='scanner.ScannerRange.xmax', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='scanner.ScannerRange.ymin', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='scanner.ScannerRange.ymax', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=164,
  serialized_end=234,
)


_SCANNERPIXELRANGE = _descriptor.Descriptor(
  name='ScannerPixelRange',
  full_name='scanner.ScannerPixelRange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='scanner.ScannerPixelRange.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='scanner.ScannerPixelRange.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=236,
  serialized_end=277,
)


_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='scanner.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='scanner.Image.data', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='scanner.Image.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='scanner.Image.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=279,
  serialized_end=331,
)


_IMAGESTACKID = _descriptor.Descriptor(
  name='ImageStackID',
  full_name='scanner.ImageStackID',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='scanner.ImageStackID.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=333,
  serialized_end=359,
)


_IMAGES = _descriptor.Descriptor(
  name='Images',
  full_name='scanner.Images',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='images', full_name='scanner.Images.images', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=361,
  serialized_end=401,
)


_IMAGESLENGTH = _descriptor.Descriptor(
  name='ImagesLength',
  full_name='scanner.ImagesLength',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='length', full_name='scanner.ImagesLength.length', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=403,
  serialized_end=433,
)


_ARRAY = _descriptor.Descriptor(
  name='Array',
  full_name='scanner.Array',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='array', full_name='scanner.Array.array', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=435,
  serialized_end=457,
)


_PIEZOPOS = _descriptor.Descriptor(
  name='PiezoPos',
  full_name='scanner.PiezoPos',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='z', full_name='scanner.PiezoPos.z', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=459,
  serialized_end=480,
)


_ZERNIKEMODES = _descriptor.Descriptor(
  name='ZernikeModes',
  full_name='scanner.ZernikeModes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='modes', full_name='scanner.ZernikeModes.modes', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='amplitudes', full_name='scanner.ZernikeModes.amplitudes', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=482,
  serialized_end=531,
)


_TIMEOUT = _descriptor.Descriptor(
  name='Timeout',
  full_name='scanner.Timeout',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timeout', full_name='scanner.Timeout.timeout', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=533,
  serialized_end=559,
)

_IMAGES.fields_by_name['images'].message_type = _IMAGE
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['ScannerResponse'] = _SCANNERRESPONSE
DESCRIPTOR.message_types_by_name['Filename'] = _FILENAME
DESCRIPTOR.message_types_by_name['ScannerStatus'] = _SCANNERSTATUS
DESCRIPTOR.message_types_by_name['ScannerRange'] = _SCANNERRANGE
DESCRIPTOR.message_types_by_name['ScannerPixelRange'] = _SCANNERPIXELRANGE
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['ImageStackID'] = _IMAGESTACKID
DESCRIPTOR.message_types_by_name['Images'] = _IMAGES
DESCRIPTOR.message_types_by_name['ImagesLength'] = _IMAGESLENGTH
DESCRIPTOR.message_types_by_name['Array'] = _ARRAY
DESCRIPTOR.message_types_by_name['PiezoPos'] = _PIEZOPOS
DESCRIPTOR.message_types_by_name['ZernikeModes'] = _ZERNIKEMODES
DESCRIPTOR.message_types_by_name['Timeout'] = _TIMEOUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Empty)
  })
_sym_db.RegisterMessage(Empty)

ScannerResponse = _reflection.GeneratedProtocolMessageType('ScannerResponse', (_message.Message,), {
  'DESCRIPTOR' : _SCANNERRESPONSE,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ScannerResponse)
  })
_sym_db.RegisterMessage(ScannerResponse)

Filename = _reflection.GeneratedProtocolMessageType('Filename', (_message.Message,), {
  'DESCRIPTOR' : _FILENAME,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Filename)
  })
_sym_db.RegisterMessage(Filename)

ScannerStatus = _reflection.GeneratedProtocolMessageType('ScannerStatus', (_message.Message,), {
  'DESCRIPTOR' : _SCANNERSTATUS,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ScannerStatus)
  })
_sym_db.RegisterMessage(ScannerStatus)

ScannerRange = _reflection.GeneratedProtocolMessageType('ScannerRange', (_message.Message,), {
  'DESCRIPTOR' : _SCANNERRANGE,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ScannerRange)
  })
_sym_db.RegisterMessage(ScannerRange)

ScannerPixelRange = _reflection.GeneratedProtocolMessageType('ScannerPixelRange', (_message.Message,), {
  'DESCRIPTOR' : _SCANNERPIXELRANGE,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ScannerPixelRange)
  })
_sym_db.RegisterMessage(ScannerPixelRange)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Image)
  })
_sym_db.RegisterMessage(Image)

ImageStackID = _reflection.GeneratedProtocolMessageType('ImageStackID', (_message.Message,), {
  'DESCRIPTOR' : _IMAGESTACKID,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ImageStackID)
  })
_sym_db.RegisterMessage(ImageStackID)

Images = _reflection.GeneratedProtocolMessageType('Images', (_message.Message,), {
  'DESCRIPTOR' : _IMAGES,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Images)
  })
_sym_db.RegisterMessage(Images)

ImagesLength = _reflection.GeneratedProtocolMessageType('ImagesLength', (_message.Message,), {
  'DESCRIPTOR' : _IMAGESLENGTH,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ImagesLength)
  })
_sym_db.RegisterMessage(ImagesLength)

Array = _reflection.GeneratedProtocolMessageType('Array', (_message.Message,), {
  'DESCRIPTOR' : _ARRAY,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Array)
  })
_sym_db.RegisterMessage(Array)

PiezoPos = _reflection.GeneratedProtocolMessageType('PiezoPos', (_message.Message,), {
  'DESCRIPTOR' : _PIEZOPOS,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.PiezoPos)
  })
_sym_db.RegisterMessage(PiezoPos)

ZernikeModes = _reflection.GeneratedProtocolMessageType('ZernikeModes', (_message.Message,), {
  'DESCRIPTOR' : _ZERNIKEMODES,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.ZernikeModes)
  })
_sym_db.RegisterMessage(ZernikeModes)

Timeout = _reflection.GeneratedProtocolMessageType('Timeout', (_message.Message,), {
  'DESCRIPTOR' : _TIMEOUT,
  '__module__' : 'doptical.api.scanner_pb2'
  # @@protoc_insertion_point(class_scope:scanner.Timeout)
  })
_sym_db.RegisterMessage(Timeout)



_SCANNER = _descriptor.ServiceDescriptor(
  name='Scanner',
  full_name='scanner.Scanner',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=562,
  serialized_end=1363,
  methods=[
  _descriptor.MethodDescriptor(
    name='StartScan',
    full_name='scanner.Scanner.StartScan',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_SCANNERRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StartLive',
    full_name='scanner.Scanner.StartLive',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopScan',
    full_name='scanner.Scanner.StopScan',
    index=2,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ClearScanImages',
    full_name='scanner.Scanner.ClearScanImages',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Capture',
    full_name='scanner.Scanner.Capture',
    index=4,
    containing_service=None,
    input_type=_TIMEOUT,
    output_type=_IMAGES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetScanRange',
    full_name='scanner.Scanner.SetScanRange',
    index=5,
    containing_service=None,
    input_type=_SCANNERRANGE,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetScanPixelRange',
    full_name='scanner.Scanner.SetScanPixelRange',
    index=6,
    containing_service=None,
    input_type=_SCANNERPIXELRANGE,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetScanRange',
    full_name='scanner.Scanner.GetScanRange',
    index=7,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_SCANNERRANGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetScanPixelRange',
    full_name='scanner.Scanner.GetScanPixelRange',
    index=8,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_SCANNERPIXELRANGE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetScanImages',
    full_name='scanner.Scanner.GetScanImages',
    index=9,
    containing_service=None,
    input_type=_IMAGESTACKID,
    output_type=_IMAGES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetScanImagesLength',
    full_name='scanner.Scanner.GetScanImagesLength',
    index=10,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_IMAGESLENGTH,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetScanImageBytes',
    full_name='scanner.Scanner.GetScanImageBytes',
    index=11,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_ARRAY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetPiezoPos',
    full_name='scanner.Scanner.SetPiezoPos',
    index=12,
    containing_service=None,
    input_type=_PIEZOPOS,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetSLMZernikeModes',
    full_name='scanner.Scanner.SetSLMZernikeModes',
    index=13,
    containing_service=None,
    input_type=_ZERNIKEMODES,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SCANNER)

DESCRIPTOR.services_by_name['Scanner'] = _SCANNER

# @@protoc_insertion_point(module_scope)
