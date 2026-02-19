* SPICE netlist for the given schematic

VCC 9 0 DC 12V
Vi 3 0 DC 0V

R1 9 6 10k
R2 6 3 1k
RC 9 5 5k
RE 4 8 200
RL 5 8 1k

Q1 5 6 4 BJT_MODEL

Cinput 3 3 1uF
Coutput 5 8 1uF

.model BJT_MODEL NPN (IS=1E-15 BF=100)

.end