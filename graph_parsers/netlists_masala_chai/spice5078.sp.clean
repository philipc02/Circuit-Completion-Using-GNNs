plaintext
* SPICE Netlist 
* Original schematic image used as reference with annotated connections

* AC Voltage Source
Vin 3 0 AC 10m

* Resistors
Rhigh 3 2 100k
RL 2 4 1

* Operational Amplifier (741C)
* Note: Ideal op-amp used for simplicity, specify appropriate model for real-world simulation
XU1 2 2 2 2 0 opamp741
.model opamp741 opamp (Rin=1Meg Rout=75 Gain=200k)

* Power Supplies for Op-Amp
Vcc 7 0 DC 15
Vee 2 0 DC -15

.end