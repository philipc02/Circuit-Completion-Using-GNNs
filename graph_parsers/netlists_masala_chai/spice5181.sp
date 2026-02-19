spice
* Operational Amplifier Circuit
* Nodes: 1 = Vin, 2 = Node connected to R1, R2, and inverting input of OpAmp
*        3 = Ground, 4 = Ground, 5 = Positive supply VCC, 6 = Vout

VCC 5 0 DC 15V
VEE 3 0 DC -15V

R1 5 2 1k
R2 2 4 10k

CBY 4 3 0.01uF

Vin 1 0 DC 0V

* Op-Amp Model
* +Input = Node 1, -Input = Node 2, Output = Node 6, VCC = Node 5, VEE = Node 3
XOPAMP 1 2 6 5 3 OPAMP_MODEL

* End of netlist