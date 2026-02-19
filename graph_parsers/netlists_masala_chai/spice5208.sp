spice
* Op-Amp Circuit
Vin in 0 DC 0V
VCC 4 2 DC 15V
VEE 2 0 DC -15V

R1 4 3 1k ; Resistor between Node 4 and Node 3
C1 in 2 10u ; Capacitor between input (Vin) and Node 2

* This is a simple op-amp model for illustration
* The actual op-amp will have multiple connections/nodes
XOPAMP 3 2 Vout 2 0 OPAMP ; Op-amp: non-inv, inv, out, vcc+, vcc-

.END