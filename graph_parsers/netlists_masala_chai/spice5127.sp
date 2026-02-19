plaintext
* Op-Amp based circuit SPICE netlist

* Voltage Sources
VCC 7 0 DC <VCC_value>
VIN vin 0 DC <Vin_value>

* Resistors
R1 5 3 <R1_value>
R2 3 6 <R2_value>
R 7 vin <R_value>
R 1 3 <R_value>
RL 6 vout <RL_value>

* Capacitors
C1 vin 1 <C1_value>
C2 vout 6 <C2_value>
C3 5 0 <C3_value>

* Op-amp model connections (Assuming a generic op-amp)
* Non-inverting input
XX1 2 3 vout VCC

* End of netlist
.end