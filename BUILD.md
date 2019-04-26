# Linux

```
python setup.py install
```

# Windows

1. modify ```fairseq/fairseq/clib/libbleu/libbleu.cpp```
   1. open ```libbleu.cpp```
   2. find the exported C++ functions ```void bleu_zero_init```, ```void bleu_one_init```, and ```void bleu_add``` within ```extern "C" {...}```
   3. add ```__declspec(dllexport)``` before each of the function declarations
2. compile: 
```
set PYTHONPATH=%PYTHONPATH%;.
python setup.py build develop --install-dir .
```
