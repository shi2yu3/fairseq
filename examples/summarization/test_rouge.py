import os
from pyrouge import Rouge155
from rouge import Rouge

hyp = open('tmp/dec/aa.1.txt').read()
ref = open('tmp/ref/aa.1.txt').read()
rouge = Rouge()
scores = rouge.get_scores(hyp, ref)
print(scores)


r = Rouge155("pyrouge/tools/ROUGE-1.5.5")
r.system_dir = 'tmp/dec'
r.model_dir = 'tmp/ref'
r.system_filename_pattern = 'aa.(\d+).txt'
r.model_filename_pattern = 'aa.#ID#.txt'
output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
output_dict
