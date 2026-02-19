plaintext
* SPICE Netlist

* Transistors
Q11 v_o1 v_a 4 NPN
Q12 v_o2 v_b 2 NPN
Q13 5 v_a v_x NPN
Q14 2 v_x v_b NPN
Q15 6 v_y v_x NPN

* Current Sources
I1 5 0 DC
I2 2 0 DC
I3 4 0 DC
I4 va 0 DC
I5 vb 0 DC

* Diode
D1 v_x Vbias D

* Resistor
R1 v_y 0 

* Nodes
Vin v_x 0 
VBias v_y 7