plaintext
* SPICE Netlist for the provided schematic

* NMOS transistor: drain=3, gate=2, source=2, body=2
M1 3 2 2 2 NMOS

* Current Source at Vcc node (4):
I1 4 3 DC 1A

* Resistors:
R1 3 2 1k
R2 2 0 1k

* Voltage Source for V_in:
Vin 1 0 DC 5V

* Output node:
* Vout is taken from the junction of R1 and R2

* Define NMOS model (placeholder values)
.model NMOS NMOS (LEVEL=1 VTO=1 KP=2u)

.end