plaintext
* SPICE Netlist

* Voltage Supply
VDD 3 0 DC 5

* Transistors
* PMOS ML: Drain=3, Gate=3, Source=5
ML 2 3 5 PMOS

* NMOS MY: Drain=2, Gate=4, Source=0
MY 2 4 0 NMOS

* NMOS MX: Drain=2, Gate=0, Source=0
MX 2 2 0 NMOS

* Voltage Inputs
VY 4 0 DC
VX 0 0 DC

* Output Node
.vo measure output