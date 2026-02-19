plaintext
* SPICE Netlist for given BJT circuit

V1 6 0 DC Vi
RS 3 6 RS
Q1 7 3 8 NPN
RL 7 4 RL

.model NPN NPN (IS=1E-16 BF=100 + VAF=100)

.control
  run
.endc
.end