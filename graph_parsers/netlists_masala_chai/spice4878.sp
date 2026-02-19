plaintext
* SPICE netlist generated from schematic

R3 3 5 1k ; Replace 1k with actual resistance
R4 5 2 1k ; Replace 1k with actual resistance

Q1 3 5 6 npn ; NPN Transistor

D1 6 2 D_zener ; Zener Diode

.model D_zener D(BV=Vz_value) ; Replace Vz_value with actual Zener voltage

.END