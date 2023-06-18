import 'package:cli/cli.dart' as cli;
import 'model.dart';
import 'package:tokenizerx/tokenizerx.dart';
import 'dart:io';

void main(List<String> arguments) {
  print('Hello world: ${cli.calculate()}!');
  final model = RWKV();
  final tokenizer = Tokenizer();

  var token = 0;
  while (true) {
    var logits = model.forward(token);
    // greedy
    var nextTokenAm = logits.max();
    var nextToken = logits.toList().indexOf(nextTokenAm);
    token = nextToken;
    stdout.write(tokenizer.decode([nextToken]));
  }

  // print available dtypes
}
