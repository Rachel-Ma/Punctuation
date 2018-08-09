# coding:utf-8

import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import pynlpir
#
# print(pynlpir.__version__)
print(dir(pynlpir))
pynlpir.open()
s = '因为我比较懒,所以我就只是修改了这句话,代码还是原博客的'
segments = pynlpir.segment(s)
for segment in segments:
    print(segment[0], '\t', segment[1])

pynlpir.close()