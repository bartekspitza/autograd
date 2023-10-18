# Autograd engine

Backpropagation implementation. This is hidden in all modern deep learning libraries behind one single method call: `backward()`, where the magic happens. I want to un-magic that magic.

## Thoughts
Originally I implemented all vector and matrix arithmetics from scratch, but then realised it was simply too slow when dealing with "larger" datasets such as mnist. So bye ~60 commits and hi `import numpy as np`.
