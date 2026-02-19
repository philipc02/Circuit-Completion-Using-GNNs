spice
*Netlist for the given differential amplifier circuit

VCC 2 0 DC 10V
Vin 1 0 DC 1V
Vb 3 0 DC 2.5V

* Resistors
RC1 2 5 10k
RC2 2 4 10k

* NMOS Transistors (Model specified separately if needed)
M1 5 7 1 1 NMOS
M2 4 8 3 3 NMOS

.model NMOS NMOS(Level=1 Vto=1 KP=0.5u)

.END