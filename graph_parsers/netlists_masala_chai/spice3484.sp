plaintext
* SPICE Netlist

VPOS 8 0 DC V+
VNEG 0 3 DC V-

I1 2 3 DC 1A

RL_left 7 8 RL 
RL_right 7 4 RL
RE_left 5 0 RE 
RE_right 5 0 RE
RO 2 3 RO

* Connections to input/outputs
VIN1 7 0 DC VII
VIN2 4 0 DC VII
VOUT 7 5

* .end statement
.end