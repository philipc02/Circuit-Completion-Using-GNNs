spice
* SPICE Netlist for given circuit

Vb 6 0 DC v_b
Iib 7 0 DC i_b
Gm 5 3 VCCS gmv1
Rpi 8 4 r_pi
Rb 4 2 r_b
Cpi 3 4 C_pi
Cmu 3 5 C_mu

* Voltage source
V1 6 3 DC

* Current source
I1 7 3 DC

* VCCS
G1 5 3 VOL=Gmv1

* Resistors
R1 8 4 R=Rpi
R2 4 2 R=Rb

* Capacitors
C1 3 4 C=Cpi
C2 3 5 C=Cmu

.end