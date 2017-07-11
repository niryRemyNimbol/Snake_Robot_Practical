# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:25:48 2017

@author: Fabian
"""

import brian2

S = Synapses (nn, nn)
S.connect(condition='if (j%64)!=0:i=j-65 or if (j%4096)-64>0:i=j-64 or if (j%64)!=63:i=j-63 or if (j%64)!=0:i=j-1 or if (j%64)!=63:i=j+1 or if (j%64)!=0: i=j+63 or if (j%4096)+64<4096:i=j+64 or if (j%64)!=63: i=j+65')