spice
* Operational Amplifier Circuit
* Components
R1 2 0 1k      ; Resistor R1 between node 2 and ground
R2 4 3 1k      ; Resistor R2 between node 4 and node 3

* Voltage Sources
VIN 2 0 DC 0   ; Input voltage source at node 2
VCC 6 0 DC 15  ; Positive supply voltage
VEE 6 0 DC -15 ; Negative supply voltage

* Op-Amp
* Op-Amp model (ideal)
XOP 3 2 5 6 6 OPAMP ; Op-amp with nodes: in- in+ out +VCC -VEE

* Model definition for ideal op-amp
.model OPAMP opamp( gain=1e5)