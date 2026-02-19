plaintext
* SPICE Netlist
M1 3 5 2 2 NMOS
M2 3 3 4 4 PMOS
RS 1 2 RS
VIN 1 0 DC Vin
VB 5 0 DC Vb
VDD 4 0 DC Vdd

*.MODEL statements for transistors should be specified as needed:
*.MODEL NMOS NMOS(Level=...)
*.MODEL PMOS PMOS(Level=...)

.END