plaintext
* SPICE netlist for the given schematic

V1 4 0 DC 5
V2 5 0 DC -5
Vsupply 1 4 DC

RS 1 3 0.5k
RB 3 0 100k

Rc 2 8  [value]
Re1 7 5 [value]
Re2 7 5 [value]

Cc 3 2 [value]
Ce 7 0 [value]

Q1 7 3 2 NPN

VCC 8 0 DC +5
VEE 5 0 DC -5

* Define component values for the resistors and capacitors 
* Replace [value] with the capacitor/resistor values as per requirement
* End of netlist

.end