#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import multiprocessing
if sys.platform == 'darwin':
    multiprocessing.set_start_method('fork')

from .sharedArr import crSharedArr
