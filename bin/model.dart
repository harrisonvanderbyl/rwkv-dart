import "utils.dart";
import 'package:ml_linalg/linalg.dart';
import 'dart:io';

class RWKV {
  MapNames vectors = <NAMES, Vector>{};
  MapNamesM matrixes = <NAMES, List<Matrix>>{};
  State state = <NAMES, List<Vector>>{};
  late int nLayers;
  late int nEmbed;
  late Vector zeros;
  late Vector ones;

  Vector forward(int token) {
    var x = vectors[NAMES.EMBED]!
        .subvector(token * nEmbed, token * nEmbed + nEmbed);

    x = layerNorm(x, 0);
    // print(x[0]);

    for (int layer = 0; layer < nLayers; layer++) {
      var xy = layerNorm(x, (4 * layer) + 2);
      var km = mix(
          state[NAMES.STATE_XY]![layer],
          xy,
          vectors[NAMES.MIX_K]!
              .subvector(layer * nEmbed, layer * nEmbed + nEmbed));

      var vm = mix(
          state[NAMES.STATE_XY]![layer],
          xy,
          vectors[NAMES.MIX_V]!
              .subvector(layer * nEmbed, layer * nEmbed + nEmbed));

      var rm = mix(
          state[NAMES.STATE_XY]![layer],
          xy,
          vectors[NAMES.MIX_R]!
              .subvector(layer * nEmbed, layer * nEmbed + nEmbed));

      state[NAMES.STATE_XY]![layer] = xy;

      var k = (matrixes[NAMES.KM]![layer] * (km)).toVector();
      var v = (matrixes[NAMES.VM]![layer] * (vm)).toVector();
      var r = sigmoid((matrixes[NAMES.RM]![layer] * (rm)).toVector());

      var aa = state[NAMES.STATE_AA]![layer];
      var bb = state[NAMES.STATE_BB]![layer];
      // var pp = state[NAMES.STATE_PP]![layer];

      var u = vectors[NAMES.BONUS]!
          .subvector(layer * nEmbed, layer * nEmbed + nEmbed);
      var w = vectors[NAMES.DECAY]!
          .subvector(layer * nEmbed, layer * nEmbed + nEmbed);
      var wr1 = aa + (u + w + k).exp() * v;
      var wr2 = bb + (u + w + k).exp();
      var wkv = (r) * wr1 / wr2;

      state[NAMES.STATE_AA]![layer] = (aa + k.exp() * v) * w.exp();
      state[NAMES.STATE_BB]![layer] = (bb + k.exp()) * w.exp();

      x = x + (matrixes[NAMES.ATT_OUT]![layer] * wkv).toVector();

      var dd = layerNorm(x, (4 * layer + 2) + 2);
      var ffnk = mix(
          state[NAMES.STATE_DD]![layer],
          dd,
          vectors[NAMES.FFN_MIX_K]!
              .subvector(layer * nEmbed, layer * nEmbed + nEmbed));

      var ffnv = mix(
          state[NAMES.STATE_DD]![layer],
          dd,
          vectors[NAMES.FFN_MIX_V]!
              .subvector(layer * nEmbed, layer * nEmbed + nEmbed));

      state[NAMES.STATE_DD]![layer] = dd;

      var fk = (matrixes[NAMES.FFN_V]![layer] *
              ((matrixes[NAMES.FFN_K]![layer] * (ffnk))
                  .toVector()
                  .mapToVector((value) => max(0, value))
                  .pow(2)))
          .toVector();

      var fv = sigmoid((matrixes[NAMES.FFN_R]![layer] * (ffnv)).toVector());

      x = x + (fk * fv);

      // var k = vectors[NAMES.MIX_K]!.subvector(
      //     layer * nEmbed * nEmbed, layer * nEmbed * nEmbed + nEmbed * nEmbed);
      // print("layer: $layer");
    }

    x = layerNorm(x, (4 * nLayers) + 2);
    x = (matrixes[NAMES.HEAD]![0] * x).toVector();

    return x;
  }

  Vector mix(Vector x, Vector y, Vector z) {
    return y * z + (ones - z) * x;
  }

  Vector sigmoid(Vector x) {
    return ones / (ones + (zeros - x).exp());
  }

  Vector layerNorm(Vector x, int offset) {
    var mean = x.mean();
    var std = sqrt((x - mean).pow(2).sum() / (nEmbed - 1));
    var y = (x - mean) / std;
    var weight = vectors[NAMES.LAYERNORMS]!
        .subvector(offset * nEmbed, offset * nEmbed + nEmbed);
    var bias = vectors[NAMES.LAYERNORMS]!
        .subvector(offset * nEmbed + nEmbed, offset * nEmbed + nEmbed * 2);
    return y * weight + bias;
  }

  RWKV({String fname = 'model.bin'}) {
    final File binfile = File(fname);
    RandomAccessFile raf = binfile.openSync(mode: FileMode.read);

    final nLayersData = raf.readSync(8);
    nLayers = bytesToUint64(nLayersData);

    final nEmbedData = raf.readSync(8);
    nEmbed = bytesToUint64(nEmbedData);

    zeros = Vector.filled(nEmbed, 0);
    ones = Vector.filled(nEmbed, 1);

    print("loading model...");

    // print('n_layers: $nLayers');
    // print('n_embed: $nEmbed');

    // 46 tensors in total
    for (var i = 0; i < 46; i++) {
      //  print a loading bar
      loadingbar(i, 46, "loading from model.bin");

      var len = getLength(i);
      var size = getSize(i, nLayers, nEmbed);
      // print('i: $i, len: $len, size: $size');
      var tensorData = raf.readSync(len * size);

      if (len == 1) {
        var tensordata =
            tensorData.buffer.asUint8List().map((e) => e / 1).toList();
        var depth = getShape(i, nEmbed)[0];
        var height = getShape(i, nEmbed)[1];

        var larerTensor = <Matrix>[];
        var znlayers = nLayers;
        if (names[i] == NAMES.HEAD) {
          znlayers = 1;
        }
        for (var j = 0; j < znlayers; j++) {
          var listOfLayer = <List<double>>[];
          for (var k = 0; k < depth; k++) {
            var row = tensordata.sublist(j * depth * height + k * height,
                j * depth * height + k * height + height);
            listOfLayer.add(row);
          }
          var tensor = Matrix.fromList(listOfLayer, dtype: DType.float32);
          larerTensor.add(tensor);
        }
        matrixes[names[i]] = larerTensor;
      } else if (len == 4) {
        var tensordata = tensorData.buffer.asFloat32List();
        var tensor = Vector.fromList(tensordata, dtype: DType.float32);
        // print(tensor[0]);
        vectors[names[i]] = tensor;
      } else if (len == 8) {
        var tensordata = tensorData.buffer.asFloat64List();
        var tensor = Vector.fromList(tensordata, dtype: DType.float32);
        // print(tensordata.sublist(0, 10));
        vectors[names[i]] = tensor;
      }
      // var vectorskeys = vectors.keys;
      // print('vectors: $vectorskeys');

      // FILL FROM HERE
    }
    for (var ii = 0; ii < 46; ii++) {
      loadingbar(ii, 46, "Dequantizing");
      for (var l = 0; l < nLayers; l++) {
        var mt = matrixes[names[ii]];
        if (mt == null) {
          if (names[ii] == NAMES.STATE_AA ||
              names[ii] == NAMES.STATE_BB ||
              names[ii] == NAMES.STATE_PP ||
              names[ii] == NAMES.STATE_XY ||
              names[ii] == NAMES.STATE_DD) {
            if (state[names[ii]] == null) {
              state[names[ii]] = [];
            }
            state[names[ii]]!.add(
                vectors[names[ii]]!.subvector(l * nEmbed, l * nEmbed + nEmbed));
          }
          continue;
        }
        if (names[ii] == NAMES.HEAD && l > 0) {
          break;
        }
        Matrix m = mt[l].transpose();
        var width = getShape(ii, nEmbed)[0];
        Vector range = vectors[qmap[names[ii]]![0]]!
            .subvector(l * width, l * width + width);
        Vector offset = vectors[qmap[names[ii]]![1]]!
            .subvector(l * width, l * width + width);

        // dequantize
        m = m.mapRows((row) => row * range + offset);
        matrixes[names[ii]]![l] = m;
      }
    }

    raf.closeSync();
  }
}
