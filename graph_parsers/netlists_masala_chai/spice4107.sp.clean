plaintext
* SPICE Netlist
VDD 7 0 DC 9V
Vi 8 0 DC

RSI 8 3 200
R1 5 7 R1_value
R2 3 2 R2_value
RD 5 4 RD_value

CC 3 5 CC_value
CL 4 6 CL_value

M1 4 5 2 2 NMOS_MODEL

* Node Voltage Definitions
.nodeset V(7)=9V V(8)=Vin_initial V(0)=0V

* MOSFET Model Definition (Example)
.model NMOS_MODEL NMOS (Level=1 KP=100u VT0=1V)

* End of Netlist