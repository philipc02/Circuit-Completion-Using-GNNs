plaintext
* SPICE Netlist for the given schematic

* Define nodes: 
* 1: input voltage (vin)
* 2: inverting input of op-amp / output of R
* 3: output node (vout)
* 0: ground

* Voltage source
Vin 1 0 DC 0 ; Input voltage source

* Resistor
R 1 2 Rvalue ; Resistor from Vin to inverting input

* Capacitors
C1 2 0 C1value ; Capacitor from inverting input to ground
C2 3 0 C2value ; Capacitor from output to ground

* Op-amp
* Note: Connect VCC and VEE to power supply nodes
XOPAMP 0 2 3 VCC VEE OPAMP_MODEL

* Power supplies
VCC VCC 0 DC Vcc_value ; Positive power supply
VEE VEE 0 DC Vee_value ; Negative power supply

* Model for the op-amp
.model OPAMP_MODEL opamp

.end