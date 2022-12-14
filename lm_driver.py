import sys
import importlib

import ex1

text = "A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat " \
       "on the rat that sat on the cat on the mat. "
nt = ex1.normalize_text(text)  # lower casing, padding punctuation with white spaces
# print(nt)
lm = ex1.Ngram_Language_Model(n=3, chars=False)
lm.build_model(nt)  # *
print(lm.get_model_dictionary())  # *
t = lm.generate(context="a cat", n=30)
for e in [
        # t,
          'a cat sat on the mat . a fat cat sat on the mat . a bat spat on the mat . a rat sat on the mat .',
          'a cat sat on the mat',
          'the rat sat on the cat']:  # *
    print('%s | %.3f' % (e, lm.evaluate(e)))
