plaintext
* SPICE Netlist for the Given Schematic

VDD VDD 0 DC 5V

* NMOS Transistor
MQ5 net9 net6 0 0 NMOS

* PMOS Transistor
MQ1 net8 net6 VDD VDD PMOS

* Current Sources
I1 net10 net8 DC 1mA
I5 net6 net9 DC 1mA

* Capacitor
C_CQbar net6 0 1uF

* Nodes identified:
* net9 - Connected to VDD, Source of Q5
* net8 - Connected to Gate of Q1, Output node
* net6 - Connected to Drain of Q5 & Gate of Q5, other terminal of I5
* net10 - Connected between drain of Q1 and one terminal of I1