* SPICE Netlist for the schematic

Vsig 4 0 DC 1V

Rsig 4 2 1MEG

* Ideal Op-amp with gain of 1
* Assuming very large input impedance and no offset voltage
Eopamp 2 0 2 3 1

Ro 2 1 100

Rl 1 5 1k

* Ground
.model gnd VSS=0