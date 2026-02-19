spice
* SPICE Netlist for the given schematic

Q1 3 4 2 NPN           ; NPN Transistor with C=3, B=4, E=2
RL 1 3 RL_value        ; Resistor RL connected between nets 1 and 3
Re 2 0 Re_value        ; Resistor Re connected between net 2 and ground

* Define the models (example values, these need to be specified or imported from a library)
.model NPN NPN (BF=100)