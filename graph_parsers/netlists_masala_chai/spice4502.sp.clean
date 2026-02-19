spice
* Inverting Amplifier Circuit
* Nodes: 1 = Vi, 2 = Inverting Input / R2, 3 = Vo, 0 = Ground

* Voltage source
V1 1 0 DC Vin

* Resistor
R1 1 2 R_value

* Capacitor
C1 2 0 C_value

* Op-amp subcircuit
* Non-inverting input connected to ground

.subckt OPAMP 2 3
* 2 = Inverting input, 3 = Output
Rin 100Meg 
Vp Vcc Vee DC 0
.model op-amp opamp(p=1m)
.ends

X1 2 0 3 OPAMP

.end