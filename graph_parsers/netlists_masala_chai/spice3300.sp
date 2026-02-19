spice
* SPICE netlist for the schematic

VDD VDD 0 DC <value>          ; Define DC voltage value for VDD
RP  VDD 3 <resistance_value>  ; Define resistance value for RP
M1  3 Vin 0 0 NMOS_MODEL      ; Instantiate the NMOS M1

G1  3 2 0 3 {-1/gm3,4}        ; Voltage controlled current source

* Model definition for NMOS
.model NMOS_MODEL NMOS (level=1 kp=<kp_value> vto=<vto_value>)

.end