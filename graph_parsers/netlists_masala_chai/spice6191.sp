* Differential Amplifier
VDD 4 0 DC 2.5
VSS 2 0 DC -2.5

* Input sources
VINPlus 8 0 DC 0
VINMinus 3 0 DC 0

* Resistor
R1 9 8 80k

* MOSFETs
M1 8 8 2 2 NCH ; Q1 (non-inverting input)
M2 3 3 2 2 NCH ; Q2 (inverting input)
M3 4 4 3 3 PCH ; Q3
M4 4 4 8 8 PCH ; Q4
M5 3 4 3 3 PCH ; Q5 (output stage)
M6 9 9 2 2 NCH ; Q6 (current mirror)
M7 2 2 2 2 NCH ; Q7 (current source)
M8 2 2 2 2 NCH ; Q8 (current source for tail)

* Output node
VOUT 3 0 DC 0

* Simulation commands
.OP
.END