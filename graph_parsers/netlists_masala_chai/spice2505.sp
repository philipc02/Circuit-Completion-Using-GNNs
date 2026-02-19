plaintext
* NMOS characteristics model as needed
M1 5 2 6 6 NMOS_MODEL

* Current source
I1 4 0 DC I_VAL  ; Define the DC value I_VAL

* Resistors
RD VDD 5 R_VAL  ; Define resistance value R_VAL
RP 6 0 R_VAL    ; Define resistance value R_VAL

* Voltage source for Vb
Vb 2 0 DC VB_VAL  ; Define the DC value VB_VAL

* Specify power supply
VDD VDD 0 DC VDD_VAL  ; Define the DC value VDD_VAL

* Output
VOUT 3 0

* Analysis commands
.control
run
.endc

*End of Netlist