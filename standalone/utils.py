#!/usr/bin/env python
from os import listdir
from os.path import isfile, join

def get_files(d, extension = ''):
    return [ f for f in listdir(d) if isfile(join(d, f)) and f.endswith(extension) ]
