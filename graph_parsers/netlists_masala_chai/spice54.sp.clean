spice
* SPICE Netlist for the given schematic

Vs 8 0 DC <value_of_Vs>
Rs 8 7 <value_of_Rs>
Ri 7 9 <value_of_Ri>
Rt 9 2 <value_of_Rt>
Ro1 10 5 <value_of_Ro1>
Ro2 3 33 <value_of_Ro2>
RL 6 4 <value_of_RL>

* Voltage node
Vout 9 4 DC 0

* Dependent current source
G1 2 10 9 0 gm

* Specify .model, .control and .end directives if necessary
*.model <model_name> NMOS/PMOS
*.control
* <commands>
*.endc

.end