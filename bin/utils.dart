import 'dart:ffi';

import 'package:ml_linalg/linalg.dart';
import 'dart:math';
export 'dart:math';
import 'dart:io';

enum NAMES {
  XBUF,
  EMBED,
  LAYERNORMS,
  STATE_XY,
  STATE_AA,
  STATE_BB,
  STATE_PP,
  STATE_DD,
  BUFFER1,
  BUFFER2,
  BUFFER3,
  BUFFER4,
  MIX_K,
  MIX_V,
  MIX_R,
  KM,
  VM,
  RM,
  KR,
  VR,
  RR,
  O1,
  O2,
  O3,
  ATT_OUT,
  ATT_OUT_R,
  ATT_OUT_O,
  FFN_MIX_K,
  FFN_MIX_V,
  FFN_K,
  FFN_V,
  FFN_R,
  FFN_KR,
  FFN_VR,
  FFN_RR,
  FFN_KO,
  FFN_VO,
  FFN_RO,
  FFN_K_BUFFER,
  FFN_V_BUFFER,
  FFN_R_BUFFER,
  DECAY,
  BONUS,
  HEAD,
  HEAD_R,
  HEAD_O
}

var names = NAMES.values;

typedef MapNames = Map<NAMES, Vector>;
typedef MapNamesM = Map<NAMES, List<Matrix>>;
typedef State = Map<NAMES, List<Vector>>;

int bytesToUint64(List<int> bytes) {
  var result = 0;
  var shift = 0;

  for (var i = 0; i < bytes.length; i++) {
    result |= bytes[i] << shift;
    shift += 8;
  }

  return result;
}

// unsigned long long sizes[46] =

int getSize(int i, int a, int b) {
  List<int> sizes = [
    b,
    50277 * b,
    4 * (a + 1) * b,
    a * b,
    a * b,
    a * b,
    a * b,
    a * b,
    b,
    50277,
    b,
    b,
    a * b,
    a * b,
    a * b,
    a * b * b,
    a * b * b,
    a * b * b,
    a * b,
    a * b,
    a * b,
    a * b,
    a * b,
    a * b,
    a * b * b,
    a * b,
    a * b,
    a * b,
    a * b,
    a * b * b * 4,
    a * b * b * 4,
    a * b * b,
    a * b,
    a * b * 4,
    a * b,
    a * b,
    a * b * 4,
    a * b,
    b,
    b,
    b * 4,
    a * b,
    a * b,
    50277 * b,
    b,
    b
  ];

  return sizes[min(i, sizes.length - 1)];
}

List<int> getShape(int i, int b) {
  List<List<int>> shapes = [
    [b],
    [50277, b],
    [b],
    [b],
    [b],
    [b],
    [b],
    [b],
    [b],
    [50277],
    [b],
    [b],
    [b],
    [b],
    [b],
    [b, b],
    [b, b],
    [b, b],
    [b],
    [b],
    [b],
    [b],
    [b],
    [b],
    [b, b],
    [b],
    [b],
    [b],
    [b],
    [b, b * 4],
    [b * 4, b],
    [b, b],
    [b],
    [b * 4],
    [b],
    [b],
    [b, 4],
    [b],
    [b],
    [b],
    [b * 4],
    [b],
    [b],
    [b, 50277],
    [b],
    [b]
  ];

  return shapes[min(i, shapes.length - 1)];
}

Map<NAMES, List<NAMES>> qmap = {
  NAMES.KM: [NAMES.KR, NAMES.O1],
  NAMES.VM: [NAMES.VR, NAMES.O2],
  NAMES.RM: [NAMES.RR, NAMES.O3],
  NAMES.ATT_OUT: [NAMES.ATT_OUT_R, NAMES.ATT_OUT_O],
  NAMES.FFN_K: [NAMES.FFN_KR, NAMES.FFN_KO],
  NAMES.FFN_V: [NAMES.FFN_VR, NAMES.FFN_VO],
  NAMES.FFN_R: [NAMES.FFN_RR, NAMES.FFN_RO],
  NAMES.HEAD: [NAMES.HEAD_R, NAMES.HEAD_O]
};

// c++ -> dart

int getLength(i) {
  // sizeof(float) = 4
  var f = 4;
  // sizeof(double) = 8
  var d = 8;
  // sizeof(uint8_t) = 1
  var g = 1;

  // the byte sizes of the tensors
  var types = [
    d,
    f,
    d,
    d,
    d,
    d,
    d,
    d,
    d,
    f,
    f,
    f,
    d,
    d,
    d,
    g,
    g,
    g,
    f,
    f,
    f,
    f,
    f,
    f,
    g,
    f,
    f,
    d,
    d,
    g,
    g,
    g,
    f,
    f,
    f,
    f,
    f,
    f,
    d,
    d,
    f,
    d,
    d,
    g,
    f,
    f
  ];

  return types[min(i, types.length - 1)];
}

void loadingbar(int i, int max, String label) {
  stdout.write('\r');
  // use block characters to print a loading bar
  var loadblock = '█' * ((i + 1) ~/ 2);
  loadblock = loadblock.padRight(max ~/ 2, '░');
  stdout.write("[$loadblock]");
  stdout.write(
      '$label: ${((i + 1) / max * 100).toStringAsFixed(2)}%'.padRight(20));

  if (i == max - 1) {
    stdout.write('\n');
  }
}
