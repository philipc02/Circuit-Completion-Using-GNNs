spice
* SPICE Netlist for the BJT Amplifier Circuit

VCC 4 0 DC 10V
Vg 3 0 AC 50mV

Q1 7 3 10 BJT_MODEL

R1 8 4 10k
R2 8 0 2.2k
RG 3 8 600
RC 4 7 3.6k
re 7 6 180
RE 6 0 820
RL 5 0 10k

C1 4 5 C1_value
C2 8 0 C2_value
C3 6 0 C3_value

.model BJT_MODEL NPN (IS=1E-15 BF=100)

* End of Netlist