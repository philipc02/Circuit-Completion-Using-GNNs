spice
* Op-amp feedback circuit
C1 3 0 C
R1 2 3 R1
R2 2 3 R2
R3 3 0 R3
XU1 4 3 2 OPAMP
VVC 3 0 DC V_C
V_VF 4 0 DC V_F

* Note: XU1 is an op-amp model with inputs (nodes 4 and 3) and output (node 2). 
* External op-amp model needs to be defined in an actual simulation.