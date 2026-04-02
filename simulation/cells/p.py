# from brian2 import NeuronGroup
# eqs_izh = '''
# dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
# du/dt = (a*(b*v - u)) / ms : 1
# I : 1
# a : 1
# b : 1
# c : 1
# d : 1
# '''
#
# P = NeuronGroup(
#     256,
#     eqs_izh,
#     threshold='v >= 30',
#     reset='v = c; u += d',
#     method='euler'
# )
#
# P.a = 0.02
# P.b = 0.2
# P.c = -65
# P.d = 8
# P.v = -65
# P.u = 'b * v'
# P.I = 8.0
#
# K = NeuronGroup(
#     64,
#     eqs_izh,
#     threshold='v >= 30',
#     reset='v = c; u += d',
#     method='euler'
# )
#
# # bursting params
# K.a = 0.02
# K.b = 0.2
# K.c = -50
# K.d = 2
# K.v = -65
# K.u = 'b * v'
# K.I = 3.0