spice
* SPICE Netlist
C1 4 2 10u  ; Capacitor C1 connected from node 4 to node 2
C2 2 3 5u   ; Capacitor C2 connected from node 2 to ground (node 3)

* Operational Amplifier with nodes
XOPAMP 2 3 Vout opamp ; Inverting input at node 2, non-inverting input grounded at node 3, output at Vout

* Ideal opamp model
.model opamp opamp (GAIN=100k)