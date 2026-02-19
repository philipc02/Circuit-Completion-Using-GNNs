plaintext
* SPICE Netlist
V1 7 0 DC 2
R1 7 5 15k
R2 3 0 20k
I5 2 4 DC 0.01
Q1 4 5 3 NPN
V2 3 0 DC -10

* NPN BJT Model (can be changed as needed)
.model NPN NPN (IS=1E-14 BF=100)

.end